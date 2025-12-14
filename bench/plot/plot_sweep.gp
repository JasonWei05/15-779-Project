# gnuplot script to plot chunk-attn sweep CSVs
# Usage (via plot_sweep.sh):
#   gnuplot -e "datafile='sweep_kv.csv'; outdir='minimal_plots'" plot_sweep.gp

if (!exists("datafile")) datafile = "sweep_kv.csv"
if (!exists("outdir")) outdir = "minimal_plots"

set datafile separator ","
set grid back
set key top left
set tics out
set border linewidth 1.2

# Columns:
# n_seqs,n_heads,n_chunks,blocks,latency_ms,tflops,compute_eff_pct,kv_gbps,kv_mem_eff_pct
col_blocks = 4
col_latency = 5
col_tflops = 6
col_ceff = 7
col_gbps = 8
col_meff = 9

# Make plots deterministic-looking
set term pngcairo size 1400,900 font ",16"

# -----------------------------------------------------------------------------
# Latency scaling (log-log)
# -----------------------------------------------------------------------------
set output sprintf("%s/latency_scaling.png", outdir)
set title "Chunk Attention (TK) - Latency vs Grid Size"
set xlabel "Grid size (blocks = heads × chunks)"
set ylabel "Latency per launch (ms)"
set logscale x 2
set logscale y 10
set format x "%g"
set format y "%.3g"
plot datafile using col_blocks:col_latency with linespoints lw 2 pt 7 ps 1.2 title "latency"

# -----------------------------------------------------------------------------
# Throughput scaling (semi-log x)
# -----------------------------------------------------------------------------
unset logscale y
set output sprintf("%s/throughput_scaling.png", outdir)
set title "Chunk Attention (TK) - Throughput vs Grid Size"
set xlabel "Grid size (blocks = heads × chunks)"
set ylabel "Throughput (TFLOPS)"
plot datafile using col_blocks:col_tflops with linespoints lw 2 pt 7 ps 1.2 title "TFLOPS"

# -----------------------------------------------------------------------------
# KV bandwidth
# -----------------------------------------------------------------------------
set output sprintf("%s/bandwidth.png", outdir)
set title "Chunk Attention (TK) - KV Bandwidth vs Grid Size"
set xlabel "Grid size (blocks = heads × chunks)"
set ylabel "KV bandwidth (GB/s)"
plot datafile using col_blocks:col_gbps with linespoints lw 2 pt 7 ps 1.2 title "KV GB/s"

# -----------------------------------------------------------------------------
# Efficiency
# -----------------------------------------------------------------------------
set output sprintf("%s/efficiency.png", outdir)
set title "Chunk Attention (TK) - Efficiency vs Grid Size"
set xlabel "Grid size (blocks = heads × chunks)"
set ylabel "Efficiency (% of peak)"
plot \
  datafile using col_blocks:col_ceff with linespoints lw 2 pt 7 ps 1.2 title "Compute eff (%)", \
  datafile using col_blocks:col_meff with linespoints lw 2 pt 5 ps 1.2 title "KV mem eff (%)"

# -----------------------------------------------------------------------------
# Combined (2-panel): latency + throughput
# -----------------------------------------------------------------------------
set output sprintf("%s/scaling_combined.png", outdir)
set multiplot layout 1,2 title "Chunk Attention (TK) Scaling"

# left: latency (log y)
set title "Latency"
set xlabel "Blocks"
set ylabel "Latency (ms)"
set logscale y 10
plot datafile using col_blocks:col_latency with linespoints lw 2 pt 7 ps 1.0 notitle

# right: throughput
unset logscale y
set title "Throughput"
set xlabel "Blocks"
set ylabel "TFLOPS"
plot datafile using col_blocks:col_tflops with linespoints lw 2 pt 7 ps 1.0 notitle

unset multiplot

# -----------------------------------------------------------------------------
# 2x2 overview: tflops, latency, bandwidth, efficiency
# -----------------------------------------------------------------------------
set output sprintf("%s/metrics_overview.png", outdir)
set multiplot layout 2,2 title "Chunk Attention (TK) Metrics Overview"

set title "Throughput"
set xlabel "Blocks"
set ylabel "TFLOPS"
plot datafile using col_blocks:col_tflops with linespoints lw 2 pt 7 ps 1.0 notitle

set title "Latency"
set xlabel "Blocks"
set ylabel "ms"
set logscale y 10
plot datafile using col_blocks:col_latency with linespoints lw 2 pt 7 ps 1.0 notitle

unset logscale y
set title "KV Bandwidth"
set xlabel "Blocks"
set ylabel "GB/s"
plot datafile using col_blocks:col_gbps with linespoints lw 2 pt 7 ps 1.0 notitle

set title "Efficiency"
set xlabel "Blocks"
set ylabel "%"
plot \
  datafile using col_blocks:col_ceff with linespoints lw 2 pt 7 ps 1.0 title "compute", \
  datafile using col_blocks:col_meff with linespoints lw 2 pt 5 ps 1.0 title "kv-mem"

unset multiplot

set output
