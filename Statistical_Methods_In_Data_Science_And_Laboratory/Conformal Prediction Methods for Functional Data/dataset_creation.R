#' ---
#' title: "Acceleration Dataset Creation"
#' subtitle: "Functional Data Preparation from Raw Phyphox Signals"
#' author: 
#'   - "Arman Feili (feili.2101835@studenti.uniroma1.it)"
#' date: "`r Sys.Date()`"
#' output:
#'   html_document:
#'     toc: true
#'     toc_depth: 3
#'     toc_float: true
#'     number_sections: true
#'     theme: flatly
#'     highlight: tango
#'     code_folding: show
#'     fig_width: 10
#'     fig_height: 6
#' ---
#'
#' # Overview
#'
#' This script processes raw accelerometer data into a functional dataset suitable for
#' conformal prediction analysis. The output is a `.rds` file containing centered and
#' raw functional curves on a common evaluation grid.
#'
#' **Data Collection:**
#'
#' | Property | Value |
#' |----------|-------|
#' | Device | Samsung Galaxy A70 |
#' | Sensor | LSM6DSM Accelerometer (STMicroelectronics) |
#' | Application | phyphox v1.2.0 |
#' | Sampling Rate | ~203 Hz |
#' | Measurement | Acceleration with gravity (m/s²) |
#'
#' **Activities Recorded:**
#'
#' | Activity | Description |
#' |----------|-------------|
#' | Standing | Stationary position |
#' | Walking (×2 sessions) | Normal pace walking |
#' | Fast Walking | Brisk walking pace |
#'
#' **Processing Pipeline:**
#'
#' 1. Trim first/last 5 seconds (handling artifacts)
#' 2. Split into 10-second non-overlapping windows (half-open intervals $[t, t+10)$)
#' 3. Resample to 200 grid points via linear interpolation
#' 4. Normalize time to $[0, 1)$
#' 5. Create centered version (subtract window mean)

#+ setup, include=FALSE
knitr::opts_chunk$set(
  echo = TRUE,
  message = FALSE,
  warning = FALSE,
  fig.align = "center",
  out.width = "100%",
  class.source = "fold-show"
)

#+ environment-setup
## --- Reproducibility ---
set.seed(2026)

## --- Required packages (base R only - no external dependencies) ---
if (!requireNamespace("knitr", quietly = TRUE)) {
  warning("Package 'knitr' not found. Install it with: install.packages('knitr')")
}

## --- Colors ---
COLORS <- list(
  Stand     = "#E69F00",  # Orange
  Walk      = "#0072B2",  # Blue
  Fast_Walk = "#009E73",  # Green
  accent    = "#D55E00",  # Vermillion
  neutral   = "#999999"   # Gray
)

#'
#' # Configuration
#'
#' The raw recordings are continuous signals lasting several minutes. We process 
#' them into fixed-length segments (windows) that can be analyzed as functional data.
#'
#' - **Trimming**: The first and last few seconds often contain artifacts from 
#'   starting/stopping the recording, so we discard them
#' - **Windowing**: We split the trimmed signal into non-overlapping 10-second chunks
#' - **Resampling**: Each window is interpolated to exactly 200 points for consistency

#+ parameters
zip_path        <- "data.zip"
data_root       <- "data"
trim_sec        <- 5
window_len_sec  <- 10
step_sec        <- 10
M_grid          <- 200
min_points_skip <- 50
min_points_warn <- 500
out_rds         <- "accel_fda_dataset.rds"

#' | Parameter | Value | Description |
#' |-----------|-------|-------------|
#' | Trim | `r trim_sec`s | Seconds removed from start/end |
#' | Window | `r window_len_sec`s | Duration of each curve |
#' | Step | `r step_sec`s | Gap between windows (non-overlapping) |
#' | Grid | `r M_grid` | Evaluation points per curve |
#' | Output | `r out_rds` | Output file name |

#+ unzip-data, results='hide'
## --- Unzip if needed ---
if (!dir.exists(data_root) && file.exists(zip_path)) {
  unzip(zipfile = zip_path, exdir = ".")
}

## --- Auto-detect data root ---
if (!dir.exists(data_root)) {
  candidate_dirs <- list.dirs(".", recursive = FALSE, full.names = FALSE)
  candidate_dirs <- setdiff(candidate_dirs, c(".", "..", "outputs", "docs"))
  valid_candidates <- sapply(candidate_dirs, function(d) {
    length(list.files(d, pattern = "Raw Data\\.csv$", recursive = TRUE)) > 0
  })
  if (sum(valid_candidates) == 1) {
    data_root <- candidate_dirs[valid_candidates]
  } else if (sum(valid_candidates) > 1) {
    stop("Multiple candidate data directories found. Please set data_root manually.")
  }
}

if (!dir.exists(data_root)) {
  stop("Data directory not found. Ensure data.zip is present or set data_root correctly.")
}

#+ helper-functions, class.source='fold-hide'
## --- Delimiter detection ---
detect_delim <- function(path) {
  header <- readLines(path, n = 1, warn = FALSE)
  if (grepl("\t", header, fixed = TRUE)) return("\t")
  if (grepl(",",  header, fixed = TRUE)) return(",")
  stop(sprintf("Cannot detect delimiter for: %s", path))
}

## --- Read phyphox CSV with strict column matching ---
read_phyphox_raw <- function(path) {
  delim <- detect_delim(path)
  
  df <- if (delim == "\t") {
    read.delim(path, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE)
  } else {
    read.csv(path, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE)
  }
  
  if (ncol(df) < 5) stop(sprintf("Unexpected schema (<5 cols): %s", path))
  
  col_names <- tolower(names(df))
  original_names <- names(df)
  
  # Find time column (REQUIRED)
  time_matches <- grep("time", col_names, value = FALSE)
  if (length(time_matches) == 0) {
    stop(sprintf("REQUIRED 'time' column not found in: %s", path))
  }
  time_col <- time_matches[1]
  
  # Find absolute acceleration column (REQUIRED)
  abs_matches <- grep("absolute", col_names, value = FALSE)
  if (length(abs_matches) == 0) {
    stop(sprintf("REQUIRED 'absolute' column not found in: %s", path))
  }
  abs_col <- abs_matches[1]
  
  if (time_col == abs_col) {
    stop(sprintf("Column matching error in: %s", path))
  }
  
  # Find x, y, z columns (optional)
  ax_col <- grep("acceleration.*x|x.*accel", col_names, value = FALSE)
  ay_col <- grep("acceleration.*y|y.*accel", col_names, value = FALSE)
  az_col <- grep("acceleration.*z|z.*accel", col_names, value = FALSE)
  ax_col <- if (length(ax_col) > 0) ax_col[1] else NA
  ay_col <- if (length(ay_col) > 0) ay_col[1] else NA
  az_col <- if (length(az_col) > 0) az_col[1] else NA
  
  out <- data.frame(
    time_s = as.numeric(df[[time_col]]),
    ax     = if (!is.na(ax_col)) as.numeric(df[[ax_col]]) else NA_real_,
    ay     = if (!is.na(ay_col)) as.numeric(df[[ay_col]]) else NA_real_,
    az     = if (!is.na(az_col)) as.numeric(df[[az_col]]) else NA_real_,
    aabs   = as.numeric(df[[abs_col]])
  )
  
  out <- out[is.finite(out$time_s) & is.finite(out$aabs), ]
  out <- out[order(out$time_s), ]
  if (any(duplicated(out$time_s))) {
    out <- out[!duplicated(out$time_s), ]
  }
  
  out
}

## --- Estimate sampling frequency ---
estimate_fs <- function(time_s) {
  dt <- diff(time_s)
  dt <- dt[is.finite(dt) & dt > 0]
  if (length(dt) < 10) return(NA_real_)
  1 / median(dt)
}

## --- Resample window to fixed grid (half-open interval [0, window_len)) ---
resample_window <- function(t, y, window_start, window_len, M) {
  # Grid without endpoint to match half-open window [ws, we)
  grid_time <- seq(0, window_len, length.out = M + 1)[-(M + 1)]
  target_t  <- window_start + grid_time
  approx(x = t, y = y, xout = target_t, method = "linear", rule = 2)$y
}

## --- Extract activity from folder name ---
activity_from_folder <- function(folder_name) {
  if (grepl("^Fast_Walk", folder_name)) return("Fast_Walk")
  if (grepl("^Walk",      folder_name)) return("Walk")
  if (grepl("^Stand",     folder_name)) return("Stand")
  return("Unknown")
}

## --- Pretty activity labels ---
pretty_activity <- function(code) {
  labels <- c(Stand = "Standing", Walk = "Walking", Fast_Walk = "Fast Walking")
  result <- labels[code]
  result[is.na(result)] <- code[is.na(result)]
  result
}

#+ locate-files
raw_files <- list.files(
  path       = data_root,
  pattern    = "Raw Data\\.csv$",
  recursive  = TRUE,
  full.names = TRUE
)
raw_files <- raw_files[!grepl("__MACOSX|\\.DS_Store", raw_files)]

if (length(raw_files) == 0) {
  stop("No Raw Data.csv files found. Check unzip/path.")
}

#+ build-curves, results='hide'
X_raw_list      <- list()
X_centered_list <- list()
meta_list       <- list()

curve_counter <- 0L
grid01 <- seq(0, 1, length.out = M_grid + 1)[-(M_grid + 1)]

skip_counters <- list(
  too_short_after_trim = 0L,
  no_valid_windows     = 0L,
  tiny_window          = 0L
)

for (path in raw_files) {
  folder_name <- basename(dirname(path))
  df <- read_phyphox_raw(path)
  fs_est <- estimate_fs(df$time_s)
  
  tmin <- min(df$time_s)
  tmax <- max(df$time_s)
  
  # Trim first/last seconds
  keep <- df$time_s >= (tmin + trim_sec) & df$time_s <= (tmax - trim_sec)
  df   <- df[keep, , drop = FALSE]
  
  if (nrow(df) < 1000) {
    skip_counters$too_short_after_trim <- skip_counters$too_short_after_trim + 1L
    next
  }
  
  tmin2 <- min(df$time_s)
  tmax2 <- max(df$time_s)
  
  starts <- seq(from = tmin2, to = (tmax2 - window_len_sec), by = step_sec)
  if (length(starts) == 0) {
    skip_counters$no_valid_windows <- skip_counters$no_valid_windows + 1L
    next
  }
  
  activity_code <- activity_from_folder(folder_name)
  activity_lbl  <- pretty_activity(activity_code)
  
  for (ws in starts) {
    we <- ws + window_len_sec
    idx <- df$time_s >= ws & df$time_s < we
    wdf <- df[idx, , drop = FALSE]
    
    if (nrow(wdf) < min_points_skip) {
      skip_counters$tiny_window <- skip_counters$tiny_window + 1L
      next
    }
    
    y_raw <- resample_window(
      t = wdf$time_s, y = wdf$aabs,
      window_start = ws, window_len = window_len_sec, M = M_grid
    )
    y_ctr <- y_raw - mean(y_raw)
    
    curve_counter <- curve_counter + 1L
    curve_id <- sprintf("curve_%04d", curve_counter)
    
    X_raw_list[[curve_id]]      <- y_raw
    X_centered_list[[curve_id]] <- y_ctr
    
    meta_list[[curve_id]] <- data.frame(
      curve_id      = curve_id,
      session_id    = folder_name,
      activity_code = activity_code,
      activity      = activity_lbl,
      source_file   = path,
      window_start  = ws,
      window_end    = we,
      n_points_raw  = nrow(wdf),
      fs_est_hz     = fs_est,
      stringsAsFactors = FALSE
    )
  }
}

#+ assemble-dataset, results='hide'
if (length(X_raw_list) == 0) {
  stop("No windows created. Check trimming/windowing parameters.")
}

X_raw      <- do.call(rbind, X_raw_list)
X_centered <- do.call(rbind, X_centered_list)
meta       <- do.call(rbind, meta_list)
rownames(X_raw) <- meta$curve_id
rownames(X_centered) <- meta$curve_id

dataset <- list(
  X_raw      = X_raw,
  X_centered = X_centered,
  grid01     = grid01,
  window_len = window_len_sec,
  step       = step_sec,
  trim_sec   = trim_sec,
  M          = M_grid,
  meta       = meta
)

saveRDS(dataset, file = out_rds)

#'
#' # Dataset Summary
#'
#' After processing the raw accelerometer recordings, we obtain a collection of 
#' **functional curves** — each representing 10 seconds of activity. Every curve 
#' is resampled to exactly `r dataset$M` equally-spaced points, allowing us to 
#' treat them as vectors of the same length. This is essential for functional 
#' data analysis methods.
#'
#' The dataset contains **`r nrow(dataset$X_raw)` curves** distributed across 
#' three activity types:

#+ dataset-summary, results='asis'
curve_counts <- as.data.frame(table(dataset$meta$activity))
names(curve_counts) <- c("Activity", "Curves")
curve_counts$Activity <- as.character(curve_counts$Activity)
total_row <- data.frame(Activity = "**Total**", Curves = sum(curve_counts$Curves))
curve_counts_display <- rbind(curve_counts, total_row)
knitr::kable(curve_counts_display, caption = "Curves by Activity")

#'
#' # Quality Assurance
#'
#' Before using the data, we run automated checks to catch potential problems:
#'
#' - **Time monotonicity**: Windows should appear in chronological order within each file
#' - **No missing values**: All curve values must be valid numbers (no NAs)
#' - **Adequate samples**: Each window should have enough raw data points (~2000 at 200Hz)
#' - **Valid acceleration range**: Values should be physically plausible (0–50 m/s²)
#' - **Unique IDs**: Every curve must have a distinct identifier
#' - **Class balance**: Activities should be reasonably balanced (min class ≥ 30% of max)

#+ qa-checks, results='asis'
qa_results <- list()

## 1) Time monotonicity
qa_results$`Time monotonicity` <- all(sapply(unique(dataset$meta$source_file), function(src) {
  sub <- dataset$meta[dataset$meta$source_file == src, ]
  all(diff(sub$window_start[order(sub$window_start)]) >= 0)
}))

## 2) No missing values
qa_results$`No missing values` <- sum(is.na(dataset$X_raw)) == 0 && sum(is.na(dataset$X_centered)) == 0

## 3) Adequate sample counts
qa_results$`Adequate samples` <- min(dataset$meta$n_points_raw) >= min_points_warn

## 4) Valid acceleration range
raw_range <- range(dataset$X_raw)
qa_results$`Valid acc. range` <- raw_range[1] >= 0 && raw_range[2] <= 50

## 5) Unique curve IDs
qa_results$`Unique IDs` <- length(unique(dataset$meta$curve_id)) == nrow(dataset$meta)

## 6) Class balance
activity_counts <- table(dataset$meta$activity_code)
qa_results$`Class balance` <- min(activity_counts) / max(activity_counts) >= 0.3

qa_df <- data.frame(
  Check = names(qa_results),
  Status = ifelse(unlist(qa_results), "PASS", "WARN"),
  stringsAsFactors = FALSE
)
knitr::kable(qa_df, caption = "Quality Assurance Results")

#'
#' # Visualizations
#'
#' The following plots help us understand the characteristics of the acceleration 
#' signals and how they differ between activities.

#+ plot-setup
activities <- unique(dataset$meta$activity_code)
activity_labels <- pretty_activity(activities)
colors <- c(Stand = COLORS$Stand, Walk = COLORS$Walk, Fast_Walk = COLORS$Fast_Walk)

#'
#' ## Raw Curves by Activity
#'
#' Each panel shows sample curves (colored) and the mean curve (dashed black) for 
#' one activity. The **absolute acceleration** combines all three axes into a single 
#' magnitude value: $|a| = \sqrt{a_x^2 + a_y^2 + a_z^2}$.
#'
#' Key observations:
#'
#' - **Standing**: Low, nearly constant acceleration (~9.8 m/s² from gravity)
#' - **Walking**: Periodic oscillations from footsteps, moderate amplitude
#' - **Fast Walking**: Similar pattern to walking but with higher peaks

#+ plot-raw-curves, fig.width=12, fig.height=10
par(mfrow = c(2, 2), mar = c(4.5, 4.5, 3.5, 1.5), family = "sans")

for (act in activities) {
  idx <- which(dataset$meta$activity_code == act)
  sample_idx <- sample(idx, min(5, length(idx)))
  
  y_range <- range(dataset$X_raw[idx, ])
  act_label <- pretty_activity(act)
  
  plot(NULL, xlim = c(0, 1), ylim = y_range,
       xlab = "Normalized time", ylab = expression("Acceleration (m/s"^2*")"),
       main = paste0("Raw curves: ", act_label, " (n = ", length(idx), ")"),
       las = 1, cex.lab = 1.1, cex.main = 1.2)
  grid(col = "gray90", lty = 1)
  
  for (i in sample_idx) {
    lines(dataset$grid01, dataset$X_raw[i, ], 
          col = adjustcolor(colors[act], 0.8), lwd = 1.8)
  }
  
  # Add mean curve
  mean_curve <- colMeans(dataset$X_raw[idx, , drop = FALSE])
  lines(dataset$grid01, mean_curve, col = "black", lwd = 2, lty = 2)
  legend("topright", legend = c("Sample", "Mean"), 
         col = c(colors[act], "black"), lwd = c(1.8, 2), lty = c(1, 2), 
         cex = 0.7, bg = "white")
}

# All activities combined
plot(NULL, xlim = c(0, 1), ylim = range(dataset$X_raw),
     xlab = "Normalized time", ylab = expression("Acceleration (m/s"^2*")"),
     main = "Sample curves: All activities",
     las = 1, cex.lab = 1.1, cex.main = 1.2)
grid(col = "gray90", lty = 1)

for (act in activities) {
  idx <- which(dataset$meta$activity_code == act)
  sample_idx <- sample(idx, min(3, length(idx)))
  for (i in sample_idx) {
    lines(dataset$grid01, dataset$X_raw[i, ], 
          col = adjustcolor(colors[act], 0.6), lwd = 1.2)
  }
}
legend("topright", legend = activity_labels, col = colors[activities], 
       lwd = 2.5, cex = 0.9, bg = "white")

#'
#' ## Centered Curves by Activity
#'
#' **Centering** subtracts each curve's mean value, removing the baseline (gravity) 
#' and focusing on the *fluctuations* around it. This is useful because:
#'
#' - It removes the constant gravity component (~9.8 m/s²)
#' - It highlights the dynamic movement patterns
#' - Curves now oscillate around zero, making comparisons easier
#'
#' The bottom-right panel compares the **mean centered curves** for each activity. 
#' Standing shows almost no variation (flat near zero), while walking activities 
#' show characteristic oscillation patterns.

#+ plot-centered-curves, fig.width=12, fig.height=10
par(mfrow = c(2, 2), mar = c(4.5, 4.5, 3.5, 1.5), family = "sans")

for (act in activities) {
  idx <- which(dataset$meta$activity_code == act)
  sample_idx <- sample(idx, min(5, length(idx)))
  
  y_range <- range(dataset$X_centered[idx, ])
  act_label <- pretty_activity(act)
  
  plot(NULL, xlim = c(0, 1), ylim = y_range,
       xlab = "Normalized time", ylab = expression("Centered acceleration (m/s"^2*")"),
       main = paste0("Centered curves: ", act_label, " (n = ", length(idx), ")"),
       las = 1, cex.lab = 1.1, cex.main = 1.2)
  grid(col = "gray90", lty = 1)
  abline(h = 0, lty = 2, col = "gray40", lwd = 1.5)
  
  for (i in sample_idx) {
    lines(dataset$grid01, dataset$X_centered[i, ], 
          col = adjustcolor(colors[act], 0.8), lwd = 1.8)
  }
}

# Mean curves comparison
y_lim_mean <- range(sapply(activities, function(act) {
  idx <- which(dataset$meta$activity_code == act)
  range(colMeans(dataset$X_centered[idx, , drop = FALSE]))
})) * 1.3

plot(NULL, xlim = c(0, 1), ylim = y_lim_mean,
     xlab = "Normalized time", ylab = expression("Centered acceleration (m/s"^2*")"),
     main = "Mean centered curves by activity",
     las = 1, cex.lab = 1.1, cex.main = 1.2)
grid(col = "gray90", lty = 1)
abline(h = 0, lty = 2, col = "gray40", lwd = 1.5)

for (act in activities) {
  idx <- which(dataset$meta$activity_code == act)
  mean_curve <- colMeans(dataset$X_centered[idx, , drop = FALSE])
  lines(dataset$grid01, mean_curve, col = colors[act], lwd = 3)
}
legend("topright", legend = activity_labels, col = colors[activities], 
       lwd = 3, cex = 0.9, bg = "white")

#'
#' ## Distribution Summaries
#'
#' These boxplots summarize key statistics across all windows, grouped by activity.
#' The notches indicate 95% confidence intervals for the median — non-overlapping 
#' notches suggest significantly different medians.
#'
#' - **Mean acceleration**: Average value over the window. Standing ≈ 9.8 m/s² (gravity only)
#' - **Variability (SD)**: How much the signal fluctuates. Walking has more variation than standing
#' - **Peak acceleration**: Maximum value reached. Fast walking produces higher peaks than regular walking

#+ plot-distributions, fig.width=12, fig.height=5
par(mfrow = c(1, 3), mar = c(5, 4.5, 3.5, 1.5), family = "sans")

## Use consistent grouping variable with explicit level ordering
grp <- factor(dataset$meta$activity_code, levels = c("Stand", "Walk", "Fast_Walk"))

# Mean acceleration by activity
mean_aabs <- rowMeans(dataset$X_raw)
boxplot(mean_aabs ~ grp,
        col = adjustcolor(colors[levels(grp)], 0.7),
        border = colors[levels(grp)],
        xaxt = "n", las = 1, notch = TRUE,
        xlab = "Activity", ylab = expression("Mean acceleration (m/s"^2*")"),
        main = "Mean acceleration per window",
        cex.lab = 1.1, cex.main = 1.2)
axis(1, at = 1:length(levels(grp)), labels = pretty_activity(levels(grp)), cex.axis = 0.95)
grid(nx = NA, ny = NULL, col = "gray90")

# Standard deviation by activity
sd_aabs <- apply(dataset$X_raw, 1, sd)
boxplot(sd_aabs ~ grp,
        col = adjustcolor(colors[levels(grp)], 0.7),
        border = colors[levels(grp)],
        xaxt = "n", las = 1, notch = TRUE,
        xlab = "Activity", ylab = expression("SD acceleration (m/s"^2*")"),
        main = "Variability per window",
        cex.lab = 1.1, cex.main = 1.2)
axis(1, at = 1:length(levels(grp)), labels = pretty_activity(levels(grp)), cex.axis = 0.95)
grid(nx = NA, ny = NULL, col = "gray90")

# Peak acceleration by activity
peak_aabs <- apply(dataset$X_raw, 1, max)
boxplot(peak_aabs ~ grp,
        col = adjustcolor(colors[levels(grp)], 0.7),
        border = colors[levels(grp)],
        xaxt = "n", las = 1, notch = TRUE,
        xlab = "Activity", ylab = expression("Peak acceleration (m/s"^2*")"),
        main = "Peak acceleration per window",
        cex.lab = 1.1, cex.main = 1.2)
axis(1, at = 1:length(levels(grp)), labels = pretty_activity(levels(grp)), cex.axis = 0.95)
grid(nx = NA, ny = NULL, col = "gray90")

#'
#' ## Activity Statistics
#'
#' Numerical summary of each activity class. These features could serve as simple 
#' discriminators between activities in a classification task.

#+ computed-stats, results='asis'
stats_table <- do.call(rbind, lapply(unique(dataset$meta$activity), function(act) {
  idx <- which(dataset$meta$activity == act)
  X_act <- dataset$X_raw[idx, , drop = FALSE]
  
  data.frame(
    Activity = act,
    N = length(idx),
    `Mean (m/s²)` = round(mean(rowMeans(X_act)), 2),
    `Peak (m/s²)` = round(max(apply(X_act, 1, max)), 1),
    `SD mean` = round(mean(apply(X_act, 1, sd)), 2),
    check.names = FALSE
  )
}))
knitr::kable(stats_table, caption = "Summary Statistics by Activity")

#'
#' ## Heatmap Overview
#'
#' These heatmaps display **all curves at once**, with each row representing one 
#' 10-second window. Curves are sorted by activity (white/black lines mark boundaries).
#'
#' - **Left (Raw)**: Brighter colors = higher acceleration. Standing appears as a 
#'   uniform band (constant gravity), while walking shows striped patterns (footsteps)
#' - **Right (Centered)**: Red/blue show positive/negative deviations from the mean. 
#'   Standing is nearly white (no variation), walking shows clear oscillatory patterns

#+ plot-heatmap, fig.width=12, fig.height=6
par(mfrow = c(1, 2), mar = c(4.5, 4.5, 3.5, 2), family = "sans")

order_idx <- order(dataset$meta$activity_code)
X_ordered <- dataset$X_centered[order_idx, ]
activity_ordered <- dataset$meta$activity_code[order_idx]
activity_counts <- table(activity_ordered)
activity_breaks <- cumsum(activity_counts)

# Raw curves heatmap
image(t(dataset$X_raw[order_idx, ]), 
      xlab = "Normalized time", ylab = "Curve index (ordered by activity)",
      main = "Raw Curves Heatmap",
      col = hcl.colors(100, "YlOrRd", rev = TRUE),
      las = 1, cex.lab = 1.1, cex.main = 1.2)
abline(h = activity_breaks[-length(activity_breaks)] / nrow(X_ordered), 
       col = "white", lwd = 2.5)
# Add activity labels on the right margin
label_positions <- c(0, activity_breaks[-length(activity_breaks)]) + activity_counts / 2
label_positions <- label_positions / nrow(X_ordered)

# Centered curves heatmap
image(t(X_ordered), 
      xlab = "Normalized time", ylab = "Curve index (ordered by activity)",
      main = "Centered Curves Heatmap",
      col = hcl.colors(100, "RdBu", rev = FALSE),
      las = 1, cex.lab = 1.1, cex.main = 1.2)
abline(h = activity_breaks[-length(activity_breaks)] / nrow(X_ordered), 
       col = "black", lwd = 2.5)

#+ session-split-helper
## Session-level train/test split function
create_session_split <- function(meta, test_fraction = 0.2, seed = 2026) {
  set.seed(seed)
  sessions <- unique(meta$session_id)
  n_test <- max(1, round(length(sessions) * test_fraction))
  test_sessions <- sample(sessions, n_test)
  is_test <- meta$session_id %in% test_sessions
  list(
    train_idx = which(!is_test),
    test_idx  = which(is_test),
    test_sessions = test_sessions,
    train_sessions = setdiff(sessions, test_sessions)
  )
}
dataset$create_session_split <- create_session_split

#'
#' # Session Structure
#'
#' Each recording session produces multiple consecutive windows. When splitting 
#' data for training and testing, it's important to keep entire sessions together 
#' (not split across train/test). Otherwise, the model might "cheat" by learning 
#' patterns specific to a particular recording rather than general activity patterns.
#'
#' The table below shows how many windows came from each session:

#+ session-structure, results='asis'
session_summary <- aggregate(curve_id ~ session_id + activity_code, 
                              data = dataset$meta, FUN = length)
names(session_summary) <- c("Session", "Activity", "Curves")
knitr::kable(session_summary, caption = "Windows per Session")

#'
#' # Dataset Components
#'
#' The saved `.rds` file contains a list with the following components. Use 
#' `readRDS("accel_fda_dataset.rds")` to load it.
#'
#' | Component | Description | Dimensions |
#' |-----------|-------------|------------|
#' | `X_raw` | Raw absolute acceleration curves (includes gravity) | n × 200 |
#' | `X_centered` | Mean-centered curves (gravity removed) | n × 200 |
#' | `grid01` | Normalized time grid $[0, 1)$ | 200 |
#' | `meta` | Metadata for each curve (activity labels, timestamps, etc.) | n rows |
#' | `create_session_split` | Helper function for proper train/test splitting | function |
#'
#' **Metadata Fields** (one row per curve):
#'
#' | Field | Description |
#' |-------|-------------|
#' | `curve_id` | Unique identifier (e.g., "curve_0001") |
#' | `session_id` | Recording session name — keep sessions together when splitting |
#' | `activity_code` | Activity label: Stand, Walk, or Fast_Walk |
#' | `activity` | Human-readable label: Standing, Walking, or Fast Walking |
#' | `window_start` | When this window starts in the original recording (seconds) |
#' | `window_end` | When this window ends (seconds) |
#' | `n_points_raw` | Number of raw samples in this window (before resampling) |
#' | `fs_est_hz` | Estimated sampling frequency of the source file |

#+ final-summary, results='hide'
## Re-save dataset with session split helper included
saveRDS(dataset, file = out_rds)

#'
#' # How to Use This Dataset
#'
#' ## Basic Loading
#'
#' Load the dataset and extract the main components:
#'
#' ```r
#' dataset <- readRDS("accel_fda_dataset.rds")
#' X <- dataset$X_centered    # n x 200 matrix (each row is one curve)
#' grid <- dataset$grid01     # time points [0, 1) where curves are evaluated
#' meta <- dataset$meta       # data frame with activity labels and metadata
#' ```
#'
#' ## Train/Test Splitting
#'
#' **Important**: Don't randomly split individual curves! Windows from the same 
#' recording session are correlated, so we must keep entire sessions together. 
#' Use the built-in helper function:
#'
#' ```r
#' # 80/20 split at the session level (not curve level)
#' split <- dataset$create_session_split(dataset$meta, test_fraction = 0.2)
#' 
#' X_train <- dataset$X_centered[split$train_idx, ]
#' X_test  <- dataset$X_centered[split$test_idx, ]
#' y_train <- dataset$meta$activity_code[split$train_idx]
#' y_test  <- dataset$meta$activity_code[split$test_idx]
#' ```
#'
#' ## Next Steps
#'
#' This dataset is ready for functional data analysis. The main analysis script 
#' (`main.R`) uses these curves for conformal prediction — building prediction 
#' sets that contain the true activity with guaranteed coverage probability.

#' ---
#' 
#' *Dataset saved to `r out_rds` (`r round(file.info(out_rds)$size / 1024, 1)` KB) on `r Sys.Date()`*

