# Define library path
.libPaths("~/R/libs")

library(dplyr)
library(readr)
library(purrr)

# =======================================
# Inputs
# Path to sim runs
rdata_path <- "./camels-spat/SimArray.Rdata"
evaluation_path <- "./camels-spat/Eval_FUSE.Rdata"
bm_path <- "camels-spat/bm_outputs_full/"


# Define periods and corresponding masks
periods <- list(
  calibration = "cal_mask",
  validation  = "val_mask",
  all         = NULL  # No mask for 'all'
)

# Define the metric to maximize for FUSE decisions
fuse_metric <- "Cal_KGE"

# define method
method = 'nse'

# define output file
output_file <- file.path("camels-spat/full_bm_skill_scores/", "nse_full_skill_scores.csv")  # change folder if needed

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


# Clean up diagnostic cols and make names unique
names(mydf) <- make.names(names(mydf))
names(mydf) <- make.unique(names(mydf))
print(names(mydf))

# Find the models with the best value of fuse_metric
best_models <- mydf %>%
  group_by(Codes) %>%
  filter(.data[[fuse_metric]] == max(.data[[fuse_metric]], na.rm = TRUE)) %>%
  slice(1) %>%  # in case of ties, take the first
  ungroup()

# Trim best_models to only the rows with Codes matching the CSV files
models_trimmed <- best_models %>%
  filter(Codes %in% csv_codes)


# Create lookup table: code → csv_file
csv_info <- data.frame(
  code = csv_codes,
  csv_file = csv_files,
  stringsAsFactors = FALSE
)

# Merge CSV file paths onto models_trimmed
models_trimmed <- models_trimmed %>%
  left_join(csv_info, by = c("Codes" = "code"))


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
# Iterate through each catchment 

# ------------------------------
# Initialize storage for skill scores
# ------------------------------
# We'll use a nested list:
# - Top level: catchment code
# - Second level: period ('calibration', 'validation', 'all')
# - Third level: benchmark metric names -> skill score values
# We'll also store latitude and longitude at the top level for each catchment
skill_scores <- list()

# ------------------------------
# Loop over all selected models
# ------------------------------
for (i in seq_len(nrow(models_trimmed))) {
  
  # ------------------------------
  # Extract simulation info
  # ------------------------------
  catchment_code <- models_trimmed$Codes[i]          # Catchment identifier
  best_decision  <- models_trimmed$ModelDecisions[i] # Best model decision for that catchment
  
  # Find the indices in the Qsim_array corresponding to this catchment and decision
  catchment_index <- which(dimnames(Qsim_array)[[3]] == catchment_code)
  decision_index  <- which(as.numeric(dimnames(Qsim_array)[[2]]) == best_decision)
  
  # Extract the Dates for the simulated catchment
  sim_time <- as.Date(dimnames(Qsim_array)[[1]])  # or use Dates directly if named
  
  # Subset Qsim_array to get the simulated streamflow time series
  cout <- Qsim_array[, decision_index, catchment_index]
  head(sim_time)
  # ------------------------------
  # Read benchmark CSV
  # ------------------------------
  csv_path <- models_trimmed$csv_file[i]
  bm_df <- read_csv(csv_path)
  
  # Keep only the dates that exist in bm_df
  mask <- sim_time %in% bm_df$time
  sim_df <- data.frame(time = sim_time[mask], Qsim = cout[mask])
  
  # Merge safely with benchmark data
  bm_df <- bm_df %>% left_join(sim_df, by = "time")
  
  # Merge the model simulation as a new column
 # bm_df <- bm_df %>% mutate(Qsim = cout)
  
  # Identify all benchmark columns (those starting with 'bm_')
  bm_columns <- grep("^bm_", names(bm_df), value = TRUE)
  
  # ------------------------------
  # Store latitude and longitude
  # ------------------------------
  # These will be included in the final data frame
  skill_scores[[catchment_code]] <- list(
    latitude  = bm_df$latitude[1],
    longitude = bm_df$longitude[1]
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
        method    = "nse"                  # Using Nash-Sutcliffe efficiency as skill metric
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
# We'll include lat/lon at catchment level, then expand each period and benchmark combination
skill_scores_long <- imap_dfr(skill_scores, function(catchment_list, catchment) {
  
  # Extract lat/lon stored at the top level of the catchment list
  lat_value <- catchment_list$latitude
  lon_value <- catchment_list$longitude
  
  # Loop over periods
  map_dfr(names(catchment_list), function(period_name) {
    
    # Skip lat/lon entries
    if(period_name %in% c("latitude","longitude")) return(NULL)
    
    benchmarks_list <- catchment_list[[period_name]]
    
    # Construct a tibble for this period containing all benchmarks
    tibble(
      catchment   = catchment,
      latitude    = lat_value,
      longitude   = lon_value,
      period      = period_name,
      benchmark   = names(benchmarks_list),
      skill_score = unlist(benchmarks_list)
    )
  })
})


# Save to CSV
dir.create(dirname(output_file), showWarnings = FALSE, recursive = TRUE)
write.csv(skill_scores_long, output_file, row.names = FALSE)

cat("Skill scores saved to", output_file, "\n")


