use hound::WavReader;
use image::{ImageBuffer, Rgb};
use rustfft::{num_complex::Complex, FftPlanner};
use std::env;
use std::thread;
use std::time::Instant;

// ============================================================
// CONFIGURATION
// ============================================================

const CHUNK_SIZE: usize = 4096; // Samples per FFT window — must be power of 2
const OVERLAP: f64 = 0.5;       // 50% overlap between consecutive windows
const NUM_WORKERS: usize = 8;   // Number of parallel FFT threads

// ============================================================
// STEP 1: READ WAV FILE
// ============================================================

/// Reads a WAV file and returns the sample rate plus all samples
/// as f32 values. Hound handles the conversion from whatever bit
/// depth the file uses (8, 16, 24, 32) into our target type.
fn read_wav(filepath: &str) -> Result<(u32, u16, Vec<f32>), Box<dyn std::error::Error>> {
    let reader = WavReader::open(filepath)?;
    let spec = reader.spec();

    println!("  Sample Rate:    {} Hz", spec.sample_rate);
    println!("  Channels:       {}", spec.channels);
    println!("  Bit Depth:      {}", spec.bits_per_sample);
    println!("  Sample Format:  {:?}", spec.sample_format);

    // Hound gives us an iterator over samples. We read them as i32
    // (which works for any integer bit depth) and normalize to [-1.0, 1.0].
    // The normalization factor depends on bit depth:
    //   16-bit → divide by 32768
    //   24-bit → divide by 8388608
    //   etc.
    let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => {
            // Float WAVs are already in [-1.0, 1.0].
            // We need a new reader since we consumed the first one.
            let reader = WavReader::open(filepath)?;
            reader
                .into_samples::<f32>()
                .map(|s| s.unwrap())
                .collect()
        }
    };

    println!("  Total Samples:  {}", samples.len());

    Ok((spec.sample_rate, spec.channels, samples))
}

// ============================================================
// STEP 2: CONVERT TO MONO
// ============================================================

/// Converts interleaved multi-channel audio (L,R,L,R,...) into
/// a single mono channel by averaging all channels per sample frame.
fn to_mono(samples: &[f32], num_channels: u16) -> Vec<f32> {
    let ch = num_channels as usize;

    if ch == 1 {
        // Already mono — return as-is.
        return samples.to_vec();
    }

    // Average all channels per frame.
    // For stereo: frame 0 = avg(samples[0], samples[1]),
    //             frame 1 = avg(samples[2], samples[3]), etc.
    samples
        .chunks_exact(ch)          // Group into frames of `ch` samples each
        .map(|frame| {             // For each frame...
            let sum: f32 = frame.iter().sum();  // Sum all channels
            sum / ch as f32                     // Average them
        })
        .collect()
}

// ============================================================
// STEP 3: CHUNK INTO OVERLAPPING WINDOWS
// ============================================================

/// Splits the mono audio into fixed-size overlapping slices.
/// Each chunk is cloned so that in-place windowing won't corrupt
/// shared data from the overlap.
fn chunk_samples(mono: &[f32], size: usize, overlap_ratio: f64) -> Vec<Vec<f32>> {
    // Hop size determines how far we advance between chunks.
    // With 50% overlap and size=4096, hop=2048.
    let hop = (size as f64 * (1.0 - overlap_ratio)) as usize;
    let mut chunks = Vec::new();

    /*
     * This saves the overlapping chunks
     * into their own vector in the chunks vector.
     * [0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15]
     *  └──────────┘                        ← window 1 (start=0)
     *           └──────────┘               ← window 2 (start=4)
     *                     └──────────┘     ← window 3 (start=8)                              
     *                          └──────────┘  ← window 4 (start=12)
     * etc...
     */
    let mut start = 0;
    while start + size <= mono.len() {
        // Clone the slice into its own Vec so windowing is safe.
        chunks.push(mono[start..start + size].to_vec());
        start += hop;
    }

    chunks
}

// ============================================================
// STEP 4: APPLY WINDOW FUNCTION
// ============================================================

/// Applies a Hamming window to each chunk in place.
///
/// The Hamming window formula is:
///   w(n) = 0.54 - 0.46 * cos(2π * n / (N - 1))
///
/// This tapers the edges of each chunk toward zero, reducing
/// spectral leakage — the smearing of energy across frequency
/// bins caused by sharp discontinuities at chunk boundaries.
/// I can parallelize this.
fn apply_hamming_window(chunks: &mut [Vec<f32>]) {
    for chunk in chunks.iter_mut() {
        let n = chunk.len() as f32;
        for (i, sample) in chunk.iter_mut().enumerate() {
            let w = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1.0)).cos();
            *sample *= w;
        }
    }
}

// ============================================================
// STEP 5 & 6: PARALLEL FFT + MAGNITUDE COMPUTATION
// ============================================================

/// Distributes FFT computation across NUM_WORKERS OS threads.
fn parallel_fft(chunks: Vec<Vec<f32>>, num_workers: usize) -> Vec<Vec<f32>> {
    let num_chunks = chunks.len();

    // Partition: split chunks into num_workers roughly equal groups,
    // pairing each chunk with its original index for reordering later.
    let indexed: Vec<(usize, Vec<f32>)> = chunks.into_iter().enumerate().collect();
    let mut partitions: Vec<Vec<(usize, Vec<f32>)>> = (0..num_workers).map(|_| Vec::new()).collect();
    for (index, chunk) in indexed.into_iter().enumerate() {
        partitions[index % num_workers].push(chunk);
    }

    // Spawn one thread per partition. Each thread owns its data — no shared state.
    let handles: Vec<_> = partitions
        .into_iter()
        .map(|partition| {
            thread::spawn(move || {
                let mut planner = FftPlanner::<f32>::new();
                let fft = planner.plan_fft_forward(CHUNK_SIZE);

                partition
                    .into_iter()
                    .map(|(index, chunk)| {
                        let mut buffer: Vec<Complex<f32>> = chunk
                            .iter()
                            .map(|&s| Complex::new(s, 0.0))
                            .collect();
                        fft.process(&mut buffer);
                        /* takes the first half
                        of the chunk because
                        FFT output is symmetric
                        this has it take the
                        positive segment to get the
                        magnitude */
                        let num_bins = CHUNK_SIZE / 2 + 1;
                        /*
                         * The mapping takes the
                         * complex value that the
                         * FFT had taken it and returns
                         * the computed value as a
                         * magnitude.
                         */
                        let magnitudes: Vec<f32> = buffer[..num_bins]
                            .iter()
                            .map(|c| c.norm())
                            .collect();
                        (index, magnitudes)
                    })
                    .collect::<Vec<(usize, Vec<f32>)>>()
                    // This is called "turbofish" syntax
                    // it tells the compiler the "concrete type"
            })
        })
        .collect();

    // Collect and reorder by original index.
    let mut spectrogram: Vec<Vec<f32>> = vec![Vec::new(); num_chunks];
    for handle in handles {
        for (index, magnitudes) in handle.join().unwrap() {
            spectrogram[index] = magnitudes;
        }
    }
    spectrogram
}

// ============================================================
// STEP 7 & 8: RENDER SPECTROGRAM AS PNG
// ============================================================

/// Renders the spectrogram as a heatmap PNG image.
///
/// Each column = one time window (one FFT result).
/// Each row = one frequency bin.
/// Color = magnitude mapped to a heat palette (black → blue → red → yellow → white).
///
/// We apply log scaling (decibels) so both quiet and loud parts
/// are visible. Without this, loud frequencies dominate and
/// everything else appears black.
fn render_spectrogram(
    spectrogram: &[Vec<f32>],
    output_path: &str,
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let time_bins = spectrogram.len();
    let freq_bins = spectrogram[0].len();

    // --- Convert to decibels and find the range ---
    // dB = 20 * log10(magnitude)
    let db_floor = -120.0_f32; // Silence floor

    let mut db_data: Vec<Vec<f32>> = Vec::with_capacity(time_bins);
    let mut global_max: f32 = db_floor;

    for col in spectrogram {
        let db_col: Vec<f32> = col
            .iter()
            .map(|&mag| {
                if mag <= 0.0 {
                    db_floor
                } else {
                    20.0 * mag.log10()
                }
            })
            .collect();

        // Track the loudest value for normalization.
        for &val in &db_col {
            if val > global_max {
                global_max = val;
            }
        }
        db_data.push(db_col);
    }

    // --- Build the image ---
    // Width = time bins, Height = frequency bins.
    // We flip the Y axis so low frequencies are at the bottom.
    let img_width = time_bins as u32;
    let img_height = freq_bins as u32;

    let img = ImageBuffer::from_fn(img_width, img_height, |x, y| {
        // Flip Y: row 0 in the image = highest frequency bin.
        let freq_idx = (freq_bins - 1) - y as usize;
        let time_idx = x as usize;

        // Normalize dB value to [0.0, 1.0] range.
        let db_val = db_data[time_idx][freq_idx];
        let normalized = ((db_val - db_floor) / (global_max - db_floor)).clamp(0.0, 1.0);

        // Map to a heat colormap:
        //   0.00 - 0.25 → black to blue
        //   0.25 - 0.50 → blue to red
        //   0.50 - 0.75 → red to yellow
        //   0.75 - 1.00 → yellow to white
        heat_color(normalized)
    });

    img.save(output_path)?;

    // Print some info about the axes for reference.
    let hop_size = (CHUNK_SIZE as f64 * (1.0 - OVERLAP)) as usize;
    let total_time = (time_bins * hop_size) as f64 / sample_rate as f64;
    let max_freq = sample_rate as f64 / 2.0;
    println!("  Image size:     {}x{} pixels", img_width, img_height);
    println!("  Time range:     0 - {:.2} seconds", total_time);
    println!("  Freq range:     0 - {} Hz", max_freq);

    Ok(())
}

/// Maps a normalized value [0.0, 1.0] to an RGB heat color.
///   0.00 → black
///   0.25 → blue
///   0.50 → red
///   0.75 → yellow
///   1.00 → white
fn heat_color(t: f32) -> Rgb<u8> {
    let (r, g, b) = match t {
        t if t < 0.25 => {
            // Black → Blue
            let s = t / 0.25;
            (0.0, 0.0, s)
        },
        t if t < 0.5 => {
            // Blue → Red
            let s = (t - 0.25) / 0.25;
            (s, 0.0, 1.0 - s)
        },
        t if t < 0.75 => {
            // Red → Yellow
            let s = (t - 0.5) / 0.25;
            (1.0, s, 0.0)
        },
        _ => {
            // Yellow → White
            let s = (t - 0.75) / 0.25;
            (1.0, 1.0, s)
        }
    };

    Rgb([
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8,
    ])
}

// ============================================================
// MAIN — TIES EVERYTHING TOGETHER
// ============================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: cargo run --release -- <input.wav> [output.png]");
        std::process::exit(1);
    }

    let input_file = &args[1];
    let output_file = if args.len() >= 3 {
        args[2].clone()
    } else {
        "spectrogram.png".to_string()
    };

    // Step 1: Read WAV
    println!("Step 1: Reading WAV file...");
    let (sample_rate, num_channels, raw_samples) = read_wav(input_file)?;

    // Step 2: Convert to mono
    println!("Step 2: Converting to mono...");
    let mono = to_mono(&raw_samples, num_channels);
    let duration = mono.len() as f64 / sample_rate as f64;
    println!("  Mono samples:   {}", mono.len());
    println!("  Duration:       {:.2} seconds", duration);

    // Step 3: Chunk into overlapping windows
    println!("Step 3: Chunking samples...");
    let mut chunks = chunk_samples(&mono, CHUNK_SIZE, OVERLAP);
    let hop = (CHUNK_SIZE as f64 * (1.0 - OVERLAP)) as usize;
    println!("  Chunks:         {} (size={}, hop={})", chunks.len(), CHUNK_SIZE, hop);

    // Step 4: Apply window function
    println!("Step 4: Applying Hamming window...");
    apply_hamming_window(&mut chunks);

    // Step 5 & 6: Parallel FFT + magnitude computation
    println!("Step 5: Running parallel FFT with {} threads...", NUM_WORKERS);
    let start = Instant::now();
    let spectrogram = parallel_fft(chunks, NUM_WORKERS);
    let elapsed = start.elapsed();
    println!("  FFT completed in {:?}", elapsed);
    println!(
        "  Spectrogram:    {} time bins x {} freq bins",
        spectrogram.len(),
        spectrogram[0].len()
    );

    // Step 7 & 8: Render and save
    println!("Step 6: Rendering spectrogram to {}...", output_file);
    render_spectrogram(&spectrogram, &output_file, sample_rate)?;

    println!("Done!");

    Ok(())
}
