# Define library path
.libPaths("~/R/libs")

library(dplyr)
library(readr)
library(purrr)

# =======================================
# This script reads the benchmark flows calculated in 01_camels-spat_bm.py, as well as the FUSE simulated timeseries and performance metrics
# to calculate the BME for each benchmark at each catchment. The output is a single csv with skill scores for each catchment and benchmark
# ===========================

# Inputs
# Path to sim runs
rdata_path <- "../camels-spat/SimArray.Rdata"
evaluation_path <- "../camels-spat/Eval_FUSE.Rdata"
bm_path <- "../camels-spat/02_results/final_bm_flows/"


# Define periods and corresponding masks
periods <- list(
  calibration = "cal_mask",
  validation  = "val_mask",
  all         = NULL  # No mask for 'all'
)

# Define the metric to rank by
fuse_metric <- "Cal_KGE"

# define method
method = 'kge'

# define output folder
output_folder <- file.path("../camels-spat/02_results/kge/", "skill_scores")

# ========================================
# Load Robjects
load(rdata_path)

load(evaluation_path)


# Rename evaluation cols
names(mydf)[3:10] <- c(
  "Cal_KGE",
  "Eval_KGE",
  "Cal_NSE",
  "Eval_NSE",
  "Cal_minusKGE",
  "Eval_minusKGE",
  "Cal_minusNSE",
  "Eval_minusNSE"
)


# ==============================
# Format model inputs
# Get list of bm catchments
csv_files <- list.files(path = bm_path, pattern = "\\.csv$", full.names = TRUE)

# Extract model decisions based on max nse
csv_codes <- gsub("_BM\\.csv$", "", basename(csv_files))


# ========================================
# Command-line argument: top rank only?
# ========================================
args <- commandArgs(trailingOnly = TRUE)
top_rank_only <- FALSE

if (length(args) > 0) {
  if (args[1] == "--top-only" || args[1] == "-t" || args[1] == "--top") {
    top_rank_only <- TRUE
    cat("\n========================================\n")
    cat("*** TOP RANK ONLY MODE ***\n")
    cat("Only processing rank 1 (best Cal_KGE models)\n")
    cat("========================================\n\n")
  }
}


# Rank all models by fuse_metric within each catchment
ranked_models <- mydf %>%
  filter(Codes %in% csv_codes) %>%
  group_by(Codes) %>%
  arrange(desc(.data[[fuse_metric]])) %>%
  mutate(rank = row_number()) %>%
  ungroup()

# Get the total number of ranks
total_ranks <- max(ranked_models$rank, na.rm = TRUE)

cat("Total number of ranks:", total_ranks, "\n")

# ========================================
# Determine which ranks to process
# ========================================
slurm_array_task_id <- Sys.getenv("SLURM_ARRAY_TASK_ID")

if (top_rank_only) {
  # Only process rank 1 (best models)
  ranks_to_process <- 1
  cat("\nProcessing ONLY rank 1 (best models by", fuse_metric, ")\n\n")
  
} else if (slurm_array_task_id != "") {
  # Running as SLURM array job - split ranks across tasks
  task_id <- as.integer(slurm_array_task_id)
  n_tasks <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_COUNT"))
  
  # Calculate which ranks this task should process
  ranks_per_task <- ceiling(total_ranks / n_tasks)
  start_rank <- (task_id - 1) * ranks_per_task + 1
  end_rank <- min(task_id * ranks_per_task, total_ranks)
  
  ranks_to_process <- start_rank:end_rank
  
  cat("\n========================================\n")
  cat("SLURM Array Job Mode\n")
  cat("Task ID:", task_id, "of", n_tasks, "\n")
  cat("Processing ranks:", start_rank, "to", end_rank, "\n")
  cat("========================================\n\n")
  
} else {
  # Not running via SLURM - process all ranks
  ranks_to_process <- 1:total_ranks
  cat("\nProcessing all ranks (not using SLURM array)\n\n")
}


# Create lookup table: code → csv_file
csv_info <- data.frame(
  code = csv_codes,
  csv_file = csv_files,
  stringsAsFactors = FALSE
)


# ========================================
# Define functions
# ---------------------------
# 1. KGE function in R
# ---------------------------
compute_kge <- function(simulated, observed) {
  # Drop NAs pairwise
  mask <- !is.na(simulated) & !is.na(observed)
  simulated <- simulated[mask]
  observed <- observed[mask]
  
  if(length(simulated) == 0 || length(observed) == 0) return(NA_real_)
  
  mean_obs <- mean(observed)
  mean_sim <- mean(simulated)
  std_obs <- sd(observed)
  std_sim <- sd(simulated)
  
  if(mean_obs == 0 || std_obs == 0) return(NA_real_)
  
  r <- if(std_obs > 0 & std_sim > 0) cor(observed, simulated) else NA_real_
  beta <- mean_sim / mean_obs
  gamma <- std_sim / std_obs
  
  if(is.na(r) | is.na(beta) | is.na(gamma)) return(NA_real_)
  
  kge <- 1 - sqrt((r - 1)^2 + (gamma - 1)^2 + (beta - 1)^2)
  return(kge)
}

# ---------------------------
# 2. Skill score function
# ---------------------------
calculate_skill_score <- function(observed, simulated, benchmark, method = "nse") {
  
  # Remove NA pairs
  mask <- !is.na(observed) & !is.na(simulated) & !is.na(benchmark)
  observed <- observed[mask]
  simulated <- simulated[mask]
  benchmark <- benchmark[mask]
  
  if(length(observed) == 0) return(NA_real_)
  
  method <- tolower(method)
  
  if(method == "nse") {
    se_sim <- sum((observed - simulated)^2)
    se_bm <- sum((observed - benchmark)^2)
    skill_score <- if(se_bm != 0) 1 - se_sim / se_bm else NA_real_
    
  } else if(method == "rmse") {
    rmse_sim <- sqrt(mean((observed - simulated)^2))
    rmse_bm <- sqrt(mean((observed - benchmark)^2))
    skill_score <- if(rmse_bm != 0) 1 - rmse_sim / rmse_bm else NA_real_
    
  } else if(method == "kge") {
    kge_sim <- compute_kge(simulated, observed)
    kge_bm  <- compute_kge(benchmark, observed)
    if(is.na(kge_sim) | is.na(kge_bm) | kge_bm == 1) return(NA_real_)
    skill_score <- (kge_sim - kge_bm) / (1 - kge_bm)
    
  } else stop("Invalid method. Choose 'nse', 'rmse', or 'kge'.")
  
  return(skill_score)
}


# ==================================
# Loop through each rank
# ==================================

# Create output directory
dir.create(output_folder, showWarnings = FALSE, recursive = TRUE)

for (current_rank in ranks_to_process) {
  
  cat("\n========================================\n")
  cat("Processing rank", current_rank, "of", total_ranks, "\n")
  cat("========================================\n")
  
  # Filter to models with the current rank
  models_at_rank <- ranked_models %>%
    filter(rank == current_rank) %>%
    left_join(csv_info, by = c("Codes" = "code"))
  
  # Skip if no models at this rank
  if(nrow(models_at_rank) == 0) {
    cat("No models found at rank", current_rank, "\n")
    next
  }
  
  cat("Number of catchments at rank", current_rank, ":", nrow(models_at_rank), "\n")
  
  # ------------------------------
  # Initialize storage for skill scores
  # ------------------------------
  skill_scores <- list()
  
  # ------------------------------
  # Loop over all models at this rank
  # ------------------------------
  for (i in seq_len(nrow(models_at_rank))) {
    
    # Progress indicator
    if (i %% 50 == 0) {
      cat("  Progress:", i, "/", nrow(models_at_rank), "catchments processed\n")
    }
    
    # ------------------------------
    # Extract simulation info
    # ------------------------------
    catchment_code <- models_at_rank$Codes[i]          # Catchment identifier
    model_decision  <- models_at_rank$ModelDecisions[i] # Model decision for that catchment
    
    # Find the indices in the Qsim_array corresponding to this catchment and decision
    catchment_index <- which(dimnames(Qsim_array)[[3]] == catchment_code)
    decision_index  <- which(as.numeric(dimnames(Qsim_array)[[2]]) == model_decision)
    
    # Check if indices are valid
    if(length(catchment_index) == 0 || length(decision_index) == 0) {
      cat("    Warning: Could not find indices for catchment", catchment_code, "\n")
      next
    }
    
    # Extract the Dates for the simulated catchment
    sim_time <- as.Date(dimnames(Qsim_array)[[1]])  # or use Dates directly if named
    
    # Subset Qsim_array to get the simulated streamflow time series
    cout <- Qsim_array[, decision_index, catchment_index]
    
    # ------------------------------
    # Read benchmark CSV
    # ------------------------------
    csv_path <- models_at_rank$csv_file[i]
    
    if(is.na(csv_path) || !file.exists(csv_path)) {
      cat("    Warning: CSV file not found for catchment", catchment_code, "\n")
      next
    }
    
    bm_df <- read_csv(csv_path, show_col_types = FALSE, name_repair = "minimal")
    
    # Remove any unnamed columns (those with empty names, NA names, or starting with ...)
    col_names <- names(bm_df)
    valid_cols <- !is.na(col_names) & col_names != "" & !grepl("^\\.\\.\\.", col_names)
    bm_df <- bm_df[, valid_cols, drop = FALSE]
    
    # Keep only the dates that exist in bm_df
    mask <- sim_time %in% bm_df$time
    sim_df <- data.frame(time = sim_time[mask], Qsim = cout[mask])
    
    # Merge safely with benchmark data
    bm_df <- bm_df %>% left_join(sim_df, by = "time")
    
    # Identify all benchmark columns (those starting with 'bm_')
    bm_columns <- grep("^bm_", names(bm_df), value = TRUE)
    
    # ------------------------------
    # Store latitude and longitude
    # ------------------------------
    skill_scores[[catchment_code]] <- list(
      latitude  = bm_df$latitude[1],
      longitude = bm_df$longitude[1],
      rank = current_rank,
      model_decision = model_decision,
      cal_kge = models_at_rank$Cal_KGE[i]
    )
    
    # ------------------------------
    # Loop over defined periods (calibration, validation, all)
    # ------------------------------
    for (period_name in names(periods)) {
      
      mask_col <- periods[[period_name]]  # Column name for the mask, if any
      
      # Subset the data to the current period if mask exists; otherwise use all data
      df_period <- if(!is.null(mask_col) && mask_col %in% names(bm_df)) {
        bm_df %>% filter(.data[[mask_col]] == TRUE)
      } else bm_df
      
      # Initialize storage for benchmarks for this period
      skill_scores[[catchment_code]][[period_name]] <- list()
      
      # ------------------------------
      # Loop over benchmark columns and calculate skill scores
      # ------------------------------
      for (bm_col in bm_columns) {
        
        skill <- calculate_skill_score(
          observed  = df_period$q_obs,       # Observed flow
          simulated = df_period$Qsim,        # Model simulated flow
          benchmark = df_period[[bm_col]],   # Benchmark flow
          method    = method                 # Using defined metric
        )
        
        # Optional: skip benchmarks that don't work for unseen data
        if(bm_col %in% c("bm_annual_mean_flow","bm_annual_median_flow")) skill <- NA_real_
        
        # Store skill score in the nested list
        skill_scores[[catchment_code]][[period_name]][[bm_col]] <- skill
      }
    }
  }
  
  # ------------------------------
  # Flatten nested list into a tidy data.frame
  # ------------------------------
  skill_scores_long <- imap_dfr(skill_scores, function(catchment_list, catchment) {
    
    # Extract metadata stored at the top level of the catchment list
    lat_value <- catchment_list$latitude
    lon_value <- catchment_list$longitude
    rank_value <- catchment_list$rank
    decision_value <- catchment_list$model_decision
    kge_value <- catchment_list$cal_kge
    
    # Loop over periods
    map_dfr(names(catchment_list), function(period_name) {
      
      # Skip metadata entries
      if(period_name %in% c("latitude","longitude","rank","model_decision","cal_kge")) return(NULL)
      
      benchmarks_list <- catchment_list[[period_name]]
      
      # Construct a tibble for this period containing all benchmarks
      tibble(
        catchment       = catchment,
        latitude        = lat_value,
        longitude       = lon_value,
        rank            = rank_value,
        model_decision  = decision_value,
        cal_kge         = kge_value,  # Cal_KGE value for this model decision at this catchment
        period          = period_name,
        benchmark       = names(benchmarks_list),
        skill_score     = unlist(benchmarks_list)
      )
    })
  })
  
  
    # Save to CSV for this rank
    output_file <- file.path(output_folder, sprintf("%s_skill_scores_rank_%03d.csv", method, current_rank))
    write.csv(skill_scores_long, output_file, row.names = FALSE)
    
    cat("Skill scores for rank", current_rank, "saved to", output_file, "\n")
    cat("Output file contains", nrow(skill_scores_long), "rows\n")
}

cat("\n========================================\n")
cat("All ranks processed successfully!\n")
cat("Output files saved in:", output_folder, "\n")
cat("========================================\n")