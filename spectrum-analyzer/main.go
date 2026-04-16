package main

import (
	"fmt"
	"log"
	"math"
	"math/cmplx"
	"os"
	"sync"
	"time"

	"github.com/serfreeman1337/go-wav"
	"gonum.org/v1/gonum/dsp/fourier"
	"gonum.org/v1/gonum/dsp/window"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette"
	"gonum.org/v1/plot/palette/moreland"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// ============================================================
// CONFIGURATION
// ============================================================

const (
	chunkSize  = 4096 // Samples per FFT window — must be power of 2
	overlap    = 0.5  // 50% overlap between consecutive windows
	numWorkers = 8    // Number of parallel FFT goroutines
)

// ============================================================
// STEP 1: READ WAV FILE
// ============================================================

// readWav opens a WAV file and returns the sample rate and raw float32 samples.
func readWav(filepath string) (uint32, uint32, []float32, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return 0, 0, nil, fmt.Errorf("could not open file: %w", err)
	}
	defer f.Close()

	reader, err := wav.NewReader(f)
	if err != nil {
		return 0, 0, nil, fmt.Errorf("could not parse WAV header: %w", err)
	}

	fmt.Printf("  Sample Rate:    %d Hz\n", reader.Frequency)
	fmt.Printf("  Channels:       %d\n", reader.NumChannels)
	fmt.Printf("  Bit Depth:      %d\n", reader.BitsPerSample)
	fmt.Printf("  Duration:       %s\n", reader.Duration)
	fmt.Printf("  Sample Count:   %d\n", reader.SampleCount)

	buf := make([]float32, reader.SampleCount)
	n, err := reader.ReadFloat(buf)
	if err != nil {
		return 0, 0, nil, fmt.Errorf("could not read samples: %w", err)
	}
	buf = buf[:n]

	return reader.Frequency, reader.NumChannels, buf, nil
}

// ============================================================
// STEP 2: CONVERT TO MONO
// ============================================================

// toMono converts interleaved multi-channel audio (L,R,L,R,...) into
// a single mono channel by averaging all channels per sample frame.
// Also converts float32 → float64 for gonum compatibility.
func toMono(samples []float32, numChannels uint32) []float64 {
	ch := int(numChannels)

	if ch == 1 {
		// Already mono — just convert to float64.
		mono := make([]float64, len(samples))
		for i, s := range samples {
			mono[i] = float64(s)
		}
		return mono
	}

	// Average all channels per frame.
	frameCount := len(samples) / ch
	mono := make([]float64, frameCount)
	for i := 0; i < frameCount; i++ {
		var sum float64
		for c := 0; c < ch; c++ {
			sum += float64(samples[i*ch+c])
		}
		mono[i] = sum / float64(ch)
	}
	return mono
}

// ============================================================
// STEP 3: CHUNK INTO OVERLAPPING WINDOWS
// ============================================================

// chunkSamples splits the mono audio into fixed-size overlapping slices.
// Each chunk is copied so that in-place windowing won't corrupt shared data.
func chunkSamples(mono []float64, size int, overlapRatio float64) [][]float64 {
	hop := int(float64(size) * (1.0 - overlapRatio)) // e.g. 4096 * 0.5 = 2048
	var chunks [][]float64

	for start := 0; start+size <= len(mono); start += hop {
		chunk := make([]float64, size)
		copy(chunk, mono[start:start+size])
		chunks = append(chunks, chunk)
	}

	return chunks
}

// ============================================================
// STEP 4: APPLY WINDOW FUNCTION
// ============================================================

// applyWindow multiplies each chunk by a Hamming window in place.
// This tapers the edges of each chunk toward zero, reducing spectral
// leakage — sharp discontinuities at chunk boundaries that would
// smear energy across frequency bins in the FFT output.
func applyWindow(chunks [][]float64) {
	for _, chunk := range chunks {
		window.Hamming(chunk) // Modifies chunk in place
	}
}

// ============================================================
// STEP 5 & 6: PARALLEL FFT + MAGNITUDE COMPUTATION
// ============================================================

// fftJob is a unit of work sent to worker goroutines.
// It carries the chunk data plus its original index so we can
// reassemble results in the correct time order.
type fftJob struct {
	index int
	chunk []float64
}

// fftResult is what each worker sends back.
type fftResult struct {
	index      int
	magnitudes []float64
}

// parallelFFT distributes FFT computation across numWorkers goroutines.
// It uses a fan-out/fan-in pattern:
//   - One goroutine sends jobs into a shared channel (fan-out)
//   - numWorkers goroutines read from that channel and compute FFTs
//   - Each worker sends results into a results channel (fan-in)
//   - The main goroutine collects and reorders the results
func parallelFFT(chunks [][]float64, numWorkers int) [][]float64 {
	jobs := make(chan fftJob, len(chunks)) // Buffered so sender doesn't block
	results := make(chan fftResult, len(chunks))
	var wg sync.WaitGroup

	// --- Spawn worker goroutines ---
	for w := 0; w < numWorkers; w++ {
		wg.Add(1) // Register this worker with the WaitGroup BEFORE launching

		go func() {
			defer wg.Done() // Signal completion when this goroutine returns

			// Each worker creates its own FFT instance to avoid sharing state.
			fft := fourier.NewFFT(chunkSize)

			// Pull jobs from the channel until it's closed.
			for job := range jobs {
				// Compute FFT — returns complex coefficients.
				coeffs := fft.Coefficients(nil, job.chunk)

				// Only keep the first half + 1 coefficients.
				// For real-valued input, the FFT output is symmetric:
				// coeffs[k] == conjugate(coeffs[N-k]), so the second
				// half is redundant. We get chunkSize/2 + 1 unique bins.
				numBins := len(coeffs)
				mags := make([]float64, numBins)

				for i, c := range coeffs {
					// cmplx.Abs returns sqrt(real² + imag²) — the magnitude
					// of each frequency bin. This tells us how strong that
					// frequency is in this time window.
					mags[i] = cmplx.Abs(c)
				}

				// Send the result back with its original index.
				results <- fftResult{index: job.index, magnitudes: mags}
			}
		}()
	}

	// --- Send all chunks as jobs ---
	for i, chunk := range chunks {
		jobs <- fftJob{index: i, chunk: chunk}
	}
	close(jobs) // Signal workers that no more jobs are coming

	// --- Wait for all workers, then close results channel ---
	// We do this in a separate goroutine so we can start collecting
	// results immediately below without deadlocking.
	go func() {
		wg.Wait()
		close(results)
	}()

	// --- Collect results and put them in time order ---
	spectrogram := make([][]float64, len(chunks))
	for result := range results {
		spectrogram[result.index] = result.magnitudes
	}

	return spectrogram
}

// ============================================================
// STEP 7 & 8: ASSEMBLE SPECTROGRAM AND PLOT
// ============================================================

// SpectrogramGrid implements the plotter.GridXYZ interface required
// by plotter.HeatMap. It maps our 2D magnitude array to a grid
// that gonum/plot can render.
type SpectrogramGrid struct {
	Data       [][]float64 // [time_index][freq_bin]
	TimeBins   int         // Number of time columns
	FreqBins   int         // Number of frequency rows
	SampleRate float64     // For labeling the frequency axis in Hz
	HopSize    int         // For labeling the time axis in seconds
}

// Dims returns the grid dimensions (columns × rows).
func (g SpectrogramGrid) Dims() (int, int) {
	return g.TimeBins, g.FreqBins
}

// Z returns the magnitude at grid position (col, row).
// We apply log scaling (decibels) so quiet and loud parts
// are both visible. Without this, loud frequencies would
// dominate and quiet ones would be invisible.
func (g SpectrogramGrid) Z(col, row int) float64 {
	val := g.Data[col][row]
	if val <= 0 {
		return -120 // Floor at -120 dB to avoid log(0)
	}
	return 20 * math.Log10(val) // Convert to decibels
}

// X returns the time in seconds for the given column.
// Offset by half a hop so the first cell's left edge starts at t=0
// rather than the cell center, which would bleed into negative time.
func (g SpectrogramGrid) X(col int) float64 {
	return float64(col*g.HopSize+g.HopSize/2) / g.SampleRate
}

// Y returns the frequency in Hz for the given row.
func (g SpectrogramGrid) Y(row int) float64 {
	return float64(row) * g.SampleRate / float64(chunkSize)
}

// plotSpectrogram renders the spectrogram as a heatmap and saves it as PNG.
func plotSpectrogram(spectrogram [][]float64, sampleRate uint32, outputPath string) error {
	hopSize := int(float64(chunkSize) * (1.0 - overlap))

	grid := SpectrogramGrid{
		Data:       spectrogram,
		TimeBins:   len(spectrogram),
		FreqBins:   len(spectrogram[0]),
		SampleRate: float64(sampleRate),
		HopSize:    hopSize,
	}

	// Create a heat palette — dark for quiet, bright for loud.
	pal := palette.Heat(12, 1)
	heatmap := plotter.NewHeatMap(grid, pal)

	// Set up the plot with labels.
	p := plot.New()
	p.Title.Text = "Spectrogram"
	p.X.Label.Text = "Time (s)"
	p.Y.Label.Text = "Frequency (Hz)"
	p.X.Min = 0
	p.Add(heatmap)

	// Add a color bar legend showing the dB scale.
	cm := moreland.BlackBody()
	cm.SetMin(heatmap.Min)
	cm.SetMax(heatmap.Max)
	colorBar := &plotter.ColorBar{ColorMap: cm}

	p.Add(colorBar)

	// Save as PNG.
	if err := p.Save(12*vg.Inch, 4*vg.Inch, outputPath); err != nil {
		return fmt.Errorf("could not save plot: %w", err)
	}

	return nil
}

// ============================================================
// MAIN — TIES EVERYTHING TOGETHER
// ============================================================

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <input.wav> [output.png]")
		os.Exit(1)
	}

	inputFile := os.Args[1]
	outputFile := "spectrogram.png"
	if len(os.Args) >= 3 {
		outputFile = os.Args[2]
	}

	// Step 1: Read WAV
	fmt.Println("Step 1: Reading WAV file...")
	sampleRate, numChannels, rawSamples, err := readWav(inputFile)
	if err != nil {
		log.Fatalf("Error reading WAV: %v", err)
	}

	// Step 2: Convert to mono
	fmt.Println("Step 2: Converting to mono...")
	mono := toMono(rawSamples, numChannels)
	fmt.Printf("  Mono samples: %d\n", len(mono))
	fmt.Printf("  Duration: %.2f seconds\n", float64(len(mono))/float64(sampleRate))

	// Step 3: Chunk into overlapping windows
	fmt.Println("Step 3: Chunking samples...")
	chunks := chunkSamples(mono, chunkSize, overlap)
	fmt.Printf("  Chunks: %d (size=%d, hop=%d)\n",
		len(chunks), chunkSize, int(float64(chunkSize)*(1.0-overlap)))

	// Step 4: Apply window function
	fmt.Println("Step 4: Applying Hamming window...")
	applyWindow(chunks)

	// Step 5 & 6: Parallel FFT + magnitude computation
	fmt.Printf("Step 5: Running parallel FFT with %d workers...\n", numWorkers)
	start := time.Now()
	spectrogram := parallelFFT(chunks, numWorkers)
	elapsed := time.Since(start)
	fmt.Printf("  FFT completed in %v\n", elapsed)
	fmt.Printf("  Spectrogram dimensions: %d time bins × %d freq bins\n",
		len(spectrogram), len(spectrogram[0]))

	// Step 7 & 8: Plot and save
	fmt.Printf("Step 6: Plotting spectrogram to %s...\n", outputFile)
	if err := plotSpectrogram(spectrogram, sampleRate, outputFile); err != nil {
		log.Fatalf("Error plotting: %v", err)
	}

	fmt.Println("Done!")
}
