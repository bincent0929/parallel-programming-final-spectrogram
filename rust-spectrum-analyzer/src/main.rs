use hound::WavReader;
use image::{ImageBuffer, Rgb};
use rustfft::{num_complex::Complex, FftPlanner};
use std::env;
use std::sync::mpsc;
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

/// A unit of work sent to worker threads.
/// Carries the chunk data plus its original index so we can
/// reassemble results in the correct time order.
struct FftJob {
    index: usize,
    chunk: Vec<f32>,
}

/// What each worker sends back.
struct FftResult {
    index: usize,
    magnitudes: Vec<f32>,
}

/// Distributes FFT computation across NUM_WORKERS OS threads.
///
/// Architecture (same fan-out/fan-in pattern as the Go version):
///
///   Main thread
///       |
///       | sends FftJobs into `job_tx`
///       v
///   [job channel] ──> Worker 0 ──┐
///                 ──> Worker 1 ──┤ each sends FftResult
///                 ──> Worker 2 ──┤ into `result_tx`
///                 ──> ...    ──┤
///                 ──> Worker 7 ──┘
///                                |
///                                v
///                          [result channel]
///                                |
///                                v
///                          Main thread collects
///                          and reorders by index
///
fn parallel_fft(chunks: Vec<Vec<f32>>, num_workers: usize) -> Vec<Vec<f32>> {
    let num_chunks = chunks.len();

    // --- Create channels ---
    // job_tx/job_rx: main thread sends jobs, workers receive them.
    // result_tx/result_rx: workers send results, main thread receives them.
    let (job_tx, job_rx) = mpsc::channel::<FftJob>();
    let (result_tx, result_rx) = mpsc::channel::<FftResult>();

    // --- Problem: mpsc::Receiver is NOT Clone ---
    // Unlike Go channels where multiple goroutines can read from one channel,
    // Rust's mpsc::Receiver can only have ONE consumer. To share it across
    // multiple worker threads, we wrap it in Arc<Mutex<>>.
    //   Arc    = Atomic Reference Counting — lets multiple threads own the value
    //   Mutex  = Mutual Exclusion — only one thread can lock and read at a time
    let job_rx = std::sync::Arc::new(std::sync::Mutex::new(job_rx));

    // --- Spawn worker threads ---
    let mut handles = Vec::new();

    for _worker_id in 0..num_workers {
        // Clone the Arc (increments reference count, doesn't copy the data)
        // and clone the result sender (mpsc allows multiple producers).
        let job_rx = std::sync::Arc::clone(&job_rx);
        let result_tx = result_tx.clone();

        let handle = thread::spawn(move || {
            // Each worker creates its own FFT planner and plan.
            // This avoids sharing mutable FFT state between threads.
            let mut planner = FftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(CHUNK_SIZE);

            loop {
                // Lock the mutex, then try to receive a job.
                // lock() blocks until no other thread holds the lock.
                // recv() blocks until a job is available or the channel closes.
                let job = {
                    let rx = job_rx.lock().unwrap();
                    rx.recv() // Returns Err when channel is closed (all senders dropped)
                };
                // Mutex is unlocked here when `rx` goes out of scope,
                // allowing other workers to receive jobs immediately.

                match job {
                    Ok(job) => {
                        // Convert real samples to complex (imaginary part = 0).
                        // rustfft operates on Complex numbers.
                        let mut buffer: Vec<Complex<f32>> = job
                            .chunk
                            .iter()
                            .map(|&s| Complex::new(s, 0.0))
                            .collect();

                        // Run the FFT in-place — buffer now contains
                        // complex frequency-domain coefficients.
                        fft.process(&mut buffer);

                        // Compute magnitudes for the first half only.
                        // For real-valued input, the FFT output is symmetric:
                        //   buffer[k] == conjugate(buffer[N-k])
                        // So we only need bins 0..N/2+1.
                        let num_bins = CHUNK_SIZE / 2 + 1;
                        let magnitudes: Vec<f32> = buffer[..num_bins]
                            .iter()
                            .map(|c| c.norm()) // norm() = sqrt(re² + im²)
                            .collect();

                        // Send result back with original index for reordering.
                        result_tx.send(FftResult {
                            index: job.index,
                            magnitudes,
                        }).unwrap();
                    }
                    Err(_) => {
                        // Channel closed — no more jobs coming. Exit the loop.
                        break;
                    }
                }
            }
            // result_tx is dropped here when the thread exits.
            // Once ALL workers exit, all clones of result_tx are dropped,
            // which causes result_rx.recv() to return Err, ending collection.
        });

        handles.push(handle);
    }

    // --- Send all chunks as jobs ---
    for (i, chunk) in chunks.into_iter().enumerate() {
        job_tx.send(FftJob { index: i, chunk }).unwrap();
    }
    // Drop the sender to close the channel.
    // Workers will finish remaining jobs, then see Err from recv() and exit.
    drop(job_tx);

    // --- Collect results ---
    // Pre-allocate the spectrogram array so we can insert at arbitrary indices.
    let mut spectrogram: Vec<Vec<f32>> = vec![Vec::new(); num_chunks];

    // Receive results until the channel closes (all workers done).
    for result in result_rx {
        spectrogram[result.index] = result.magnitudes;
    }

    // --- Wait for all threads to finish ---
    // join() ensures we don't return until every thread has exited cleanly.
    for handle in handles {
        handle.join().unwrap();
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
    let (r, g, b) = if t < 0.25 {
        // Black → Blue
        let s = t / 0.25;
        (0.0, 0.0, s)
    } else if t < 0.5 {
        // Blue → Red
        let s = (t - 0.25) / 0.25;
        (s, 0.0, 1.0 - s)
    } else if t < 0.75 {
        // Red → Yellow
        let s = (t - 0.5) / 0.25;
        (1.0, s, 0.0)
    } else {
        // Yellow → White
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0, s)
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
