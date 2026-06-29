library(shiny)
library(shinyjs)
library(DT)
library(reticulate)
use_condaenv("/Users/jasperzzz/miniconda3/envs/hu_ssc", required = TRUE)
library(jsonlite)
library(ggplot2)
library(gridExtra)
library(scales)

# Allow large CSV uploads (e.g. 100MB)
options(shiny.maxRequestSize = 100 * 1024^2)

# Ensure working directory is the app directory
if (file.exists("app.R")) {
  # already correct
} else {
  script_dir <- tryCatch(dirname(normalizePath(sys.frames()[[1]]$ofile)), error = function(e) NULL)
  if (!is.null(script_dir) && dir.exists(script_dir)) setwd(script_dir)
}

# -----------------------------
# Global setup
# -----------------------------

source_python("predict_model.py")
meta <- jsonlite::read_json("model_meta.json", simplifyVector = TRUE)
model_candidates <- c(
  "/home/UT_shared/result/500_embedding_duration_10_deephit_tabtransformer_shared_node_epoch50.pt",
  "end-lr0.0001-wd0.01-eta0.8-alpha0.2-bs256-drop0.1-epochs100-dur30-embed500.pt"
)
model_candidates <- normalizePath(model_candidates[file.exists(model_candidates)], mustWork = FALSE)
MODEL_PATH <- model_candidates[1]
if (is.na(MODEL_PATH) || MODEL_PATH == "") stop("No model file found; update MODEL_PATH.")

load_model(MODEL_PATH)

cat_features <- meta$categ_idx
cont_features <- meta$cont_idx
feature_names <- meta$feature_names
# Index by the actual 0-based metadata indices (robust even if not contiguous)
cat_names <- feature_names[cat_features + 1L]
cont_names <- feature_names[cont_features + 1L]
duration_labels <- meta$duration_labels
risk_names <- meta$risk_names

bin_year_labels <- sapply(0:9, function(i) {
  yr <- round(i / 9 * 14.9, 1)
  paste0("bin_", i, "\n(", yr, " yr)")
})

`%||%` <- function(a, b) if (!is.null(a) && length(a) > 0 && !anyNA(a)) a else b

# Shortened label for display; full name in tooltip
truncate_label <- function(x, max_len = 28) {
  ifelse(nchar(x) > max_len, paste0(substr(x, 1, max_len - 3), "..."), x)
}


example_path <- "../data/df_imputed_synthetic.csv"
if (file.exists(example_path)) {
  tmp <- read.csv(example_path, nrows = 5, check.names = FALSE)
  if (all(feature_names %in% names(tmp))) {
    example_defaults <- tmp[1, feature_names]
  } else {
    example_defaults <- setNames(as.list(rep(0, length(feature_names))), feature_names)
  }
} else {
  example_defaults <- setNames(as.list(rep(0, length(feature_names))), feature_names)
}

# -----------------------------
# UI
# -----------------------------

ui <- navbarPage(
  title = "AF & All-Cause Mortality Risk Prediction",
  id = "main_nav",
  header = useShinyjs(),
  tabPanel(
    "Predict",
    fluidPage(
      fluidRow(
        # ---------- Left panel: Data Entry ----------
        column(
          width = 4,
          wellPanel(
            style = "background: #f8f9fa; border: 1px solid #dee2e6;",
            h4("Data Entry", style = "margin-top:0;font-weight:bold"),
            tags$hr(),
            fileInput("csv_file", "Upload CSV (one row per individual)", accept = ".csv"),
            fluidRow(
              column(4, numericInput("cursor_index", "Current row:", value = 1, min = 1, max = 1, step = 1)),
              column(4, actionButton("cursor_up", "▲", title = "Previous row", style = "margin-top: 25px;")),
              column(4, actionButton("cursor_down", "▼", title = "Next row", style = "margin-top: 25px;"))
            ),
            helpText("Select a row to view/edit its features and run prediction for that individual. Use arrows or type a number."),
            tags$hr(),
            h5("Features (all 78 — editable)", style = "margin-top:8px;font-weight:bold"),
            helpText("Values are seeded from the selected CSV row; any missing value is filled with 0. Label truncated; hover for full name. Integer codes for categorical.", style = "font-size:0.85em;"),
            div(
              style = "max-height: 780px; overflow-y: auto; overflow-x: hidden; border: 1px solid #e0e0e0; border-radius: 4px; padding: 6px; background: #ffffff;",
              uiOutput("manual_inputs_ui")
            )
          )
        ),
        # ---------- Middle panel: Prediction Output ----------
        column(
          width = 4,
          wellPanel(
            style = "background: #f8f9fa; border: 1px solid #dee2e6;",
            h4("Prediction Output", style = "margin-top:0;font-weight:bold"),
            selectInput("risk_head", "Outcome:", choices = risk_names, selected = "AF"),
            radioButtons("xaxis_mode", "X-axis:", choices = c("Bins" = "bin", "Years" = "year"), selected = "bin", inline = TRUE),
            actionButton("predict", "Run Prediction", class = "btn-primary btn-block", style = "margin-bottom:12px;"),
            fluidRow(
              column(6, downloadButton("download_plot_png", "Save curves (PNG)", style = "width: 100%; padding: 8px 12px; font-size: 14px;")),
              column(6, downloadButton("download_table_csv", "Save table (CSV)", style = "width: 100%; padding: 8px 12px; font-size: 14px;"))
            ),
            tags$div(style = "margin-bottom: 10px;"),

            # CP upper bound box — shown for whichever outcome is selected
            uiOutput("cp_upper_bound_ui"),

            h5("Event probability", style = "font-weight:bold; margin-top:8px;"),
            plotOutput("predictionPlot_prob", height = "200px"),
            h5("Survival", style = "font-weight:bold; margin-top:8px;"),
            plotOutput("predictionPlot_surv", height = "200px"),
            DTOutput("predictionTable")
          )
        ),
        # ---------- Right panel: Feature Importance ----------
        column(
          width = 4,
          wellPanel(
            style = "background: #f8f9fa; border: 1px solid #dee2e6;",
            h4("Feature Importance", style = "margin-top:0;font-weight:bold"),
            helpText("Test-case level: which features most influence this individual's prediction.", style = "font-size:0.85em;"),
            uiOutput("importance_scope_note"),
            selectInput("importance_time_bin", "Importance metric:", 
              choices = setNames(
                as.character(c(-2, -1, seq_along(duration_labels) - 1L)), 
                c("Overall (cumulative risk)", "Last bin", if (length(duration_labels) > 0) duration_labels else paste0("bin_", seq_len(10) - 1))
              ), selected = "-2"),
            actionButton("compute_importance", "Compute Importance", class = "btn-primary btn-block", style = "margin-bottom:10px;"),
            downloadButton("download_importance_csv", "Save importance (CSV)", style = "width: 100%; padding: 8px 12px; font-size: 14px; margin-bottom:10px;"),
            h5("Top features (sensitivity)", style = "font-weight:bold; margin-top:10px;"),
            plotOutput("importancePlot", height = "380px"),
            DTOutput("importanceTable")
          )
        )
      )
    )
  ),
  tabPanel(
    "Help",
    fluidPage(
      wellPanel(
        h4("AF & All-Cause Mortality Risk Prediction - Help", style = "margin-top:0;font-weight:bold"),
        p("This application estimates short-term risk patterns for two outcomes: AF (atrial fibrillation) and All cause death. It uses a TabTransformer + DeepHit model and returns per-bin event probability and derived survival curves for the selected outcome."),
        tags$hr(),
        h5("Model description", style = "font-weight:bold"),
        tags$ul(
          tags$li("Inputs: 78 features in fixed model order (61 categorical + 17 continuous)."),
          tags$li("Architecture: TabTransformer encoder with DeepHit-style risk-specific output heads."),
          tags$li("Time output: 10 discrete bins. For each outcome, the model outputs event probability by bin."),
          tags$li("Survival is computed as 1 minus the cumulative event probability across bins.")
        ),
        tags$hr(),
        h5("How to use this 3-panel interface", style = "font-weight:bold"),
        h6("Panel 1: Data Entry", style = "font-weight:bold"),
        tags$ul(
          tags$li("Upload a CSV (one row per individual) with the 78 feature columns. Select Current row to load that individual's values."),
          tags$li("All 78 features are shown as editable fields in a scrollable list, seeded from the selected row. You can edit any value before running the prediction."),
          tags$li("Missing data is handled automatically: any blank/NA cell, or any required column absent from the CSV, is filled with 0, and a notification tells you how many values were filled."),
          tags$li("Labels are shortened and show the full name on hover. Categorical features use integer codes.")
        ),
        h6("Panel 2: Prediction Output", style = "font-weight:bold"),
        tags$ul(
          tags$li("Choose outcome: AF (default) or All cause death."),
          tags$li("Click Run Prediction to generate outputs."),
          tags$li("Both event probability and survival are shown: two plots and one table with columns Duration, Event probability, and Survival."),
          tags$li("Run Prediction predicts for the current record only; change the row (or edit fields) and click Run Prediction again for another individual.")
        ),
        h6("Panel 3: Feature Importance", style = "font-weight:bold"),
        tags$ul(
          tags$li("Test-case level: shows which features most influence this individual's prediction."),
          tags$li("Uses the same patient data as the last Run Prediction (the current record's editable fields)."),
          tags$li("Importance metric: Overall (cumulative risk, default), Last bin, or a specific bin."),
          tags$li(tags$strong("Compute Importance does NOT run automatically after Run Prediction."), " You must click the Compute Importance button manually each time you want importance results."),
          tags$li(tags$strong("Importance is outcome-specific."), " To get importance for AF: select AF in the Outcome selector (Panel 2) and click Compute Importance. To get importance for All cause death: switch the Outcome selector to All cause death and click Compute Importance again. Each outcome produces different importance scores."),
          tags$li("Importance resets automatically after each Run Prediction — you must click Compute Importance again to see updated results."),
          tags$li("Requires reference data: a CSV with 2+ rows (other rows used as reference), or example_input.csv as a fallback."),
          tags$li("Save importance (CSV): downloads Feature and Importance columns.")
        ),
        tags$hr(),
        h5("Save results", style = "font-weight:bold"),
        tags$ul(
          tags$li("Save curves (PNG): downloads both event probability and survival curves in one image."),
          tags$li("Save table (CSV): downloads the table with Duration, Event probability, and Survival columns."),
          tags$li("File names include the selected outcome.")
        ),
        tags$hr(),
        h5("Notes", style = "font-weight:bold"),
        tags$ul(
          tags$li("CSV upload limit is set to 100 MB."),
          tags$li("Categorical values should use integer codes aligned with the model training schema."),
          tags$li("If Current row is out of range (e.g., greater than the number of rows in the file), the app shows a warning and adjusts to a valid row."),
          tags$li("Feature importance uses permutation (sensitivity): each feature is replaced with a random value from the reference; importance = |change in prediction|."),
          tags$li("Importance is computed for all 78 features of the current record.")
        )
      )
    )
  )
)

# -----------------------------
# Server
# -----------------------------

server <- function(input, output, session) {

  # Data Entry: all 78 features, editable (seeded from selected CSV row)
  output$manual_inputs_ui <- renderUI({
    top_cat <- cat_names
    top_cont <- cont_names
    tagList(
      lapply(top_cat, function(feat) {
        fluidRow(
          column(
            width = 6,
            tags$label(
              `for` = paste0("feat_", feat),
              title = feat,
              style = "font-size: 0.9em; padding-top: 6px;",
              truncate_label(feat)
            ),
          ),
          column(
            width = 6,
            numericInput(
              inputId = paste0("feat_", feat),
              label = NULL,
              value = as.integer(example_defaults[[feat]] %||% 0),
              min = 0,
              step = 1
            )
          )
        )
      }),
      tags$hr(style = "margin: 8px 0;"),
      lapply(top_cont, function(feat) {
        fluidRow(
          column(
            width = 6,
            tags$label(
              `for` = paste0("feat_", feat),
              title = feat,
              style = "font-size: 0.9em; padding-top: 6px;",
              truncate_label(feat)
            ),
          ),
          column(
            width = 6,
            numericInput(
              inputId = paste0("feat_", feat),
              label = NULL,
              value = as.numeric(example_defaults[[feat]] %||% 0),
              step = 0.1
            )
          )
        )
      })
    )
  })

  # ── Server-side source of truth for the 78 feature values ──────────────────
  # Predictions & importance read THIS named list, not the browser inputs, so
  # that loading a CSV row / switching rows and immediately running a prediction
  # cannot read stale input values (updateNumericInput round-trips to the client
  # asynchronously). sync_csv_to_inputs() writes here synchronously.
  .init_feature_vals <- setNames(
    lapply(feature_names, function(f) {
      v <- example_defaults[[f]] %||% 0
      if (f %in% cat_names) as.integer(v) else as.numeric(v)
    }),
    feature_names
  )
  feature_vals <- reactiveVal(.init_feature_vals)

  # Capture user edits to the editable fields back into the source of truth.
  # Depends only on the inputs (feature_vals read is isolated) so it never
  # re-triggers itself; the echo of sync's updateNumericInput calls is a no-op
  # here because the values already match feature_vals.
  observe({
    cur <- isolate(feature_vals())
    if (is.null(cur)) return()
    newvals <- cur
    changed <- FALSE
    for (feat in feature_names) {
      iv <- input[[paste0("feat_", feat)]]
      if (is.null(iv)) next
      iv <- if (feat %in% cat_names) as.integer(iv) else as.numeric(iv)
      if (length(iv) == 0 || is.na(iv)) next
      if (!isTRUE(iv == cur[[feat]])) { newvals[[feat]] <- iv; changed <- TRUE }
    }
    if (changed) feature_vals(newvals)
  })

  manualData <- reactive({
    vals <- feature_vals()
    df <- as.data.frame(
      lapply(feature_names, function(f) vals[[f]]),
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
    names(df) <- feature_names
    df
  })

  # One-time CSV storage: parsed when file is uploaded, cleared on mode switch
  csv_store <- reactiveVal(NULL)

  csv_data <- reactive({
    req(input$csv_file)
    df <- tryCatch(
      read.csv(input$csv_file$datapath, check.names = FALSE, stringsAsFactors = FALSE),
      error = function(e) NULL
    )
    shiny::validate(need(!is.null(df), "Failed to read CSV"))
    # Missing columns are no longer a hard error: add them as NA so they get
    # filled with 0 downstream, and tell the user which ones were absent.
    missing_cols <- setdiff(feature_names, names(df))
    if (length(missing_cols) > 0) {
      for (mc in missing_cols) df[[mc]] <- NA
      showNotification(
        sprintf("CSV is missing %d required column(s); they will be filled with 0: %s",
                length(missing_cols), paste(missing_cols, collapse = ", ")),
        type = "warning", duration = 8
      )
    }
    csv_store(df)  # store for prediction (one-time read)
    df
  })

  # Cursor index, validated against CSV row count
  cursor_index_valid <- reactive({
    req(input$csv_file)
    df <- csv_data()
    n <- nrow(df)
    cur <- suppressWarnings(as.integer(input$cursor_index))
    if (is.na(cur) || cur < 1) return(1L)
    if (cur > n) return(NULL)  # out of range
    cur
  })

  # Update cursor_index max and sync features when CSV loads
  observeEvent(csv_data(), {
    df <- csv_data()
    n <- nrow(df)
    cur <- suppressWarnings(as.integer(input$cursor_index %||% 1))
    if (is.na(cur) || cur < 1) cur <- 1L
    val <- min(max(1L, cur), n)
    updateNumericInput(session, "cursor_index", max = n, min = 1, value = val)
    sync_csv_to_inputs()  # Refresh feature display for current row
  }, ignoreInit = TRUE)

  # cursor_row(): single 1-row data.frame from the editable feature fields
  cursor_row <- reactive({
    # The 78 editable fields are the source of truth (seeded from the CSV row,
    # then possibly edited by the user).
    row <- manualData()
    names(row) <- feature_names
    row[, cat_names] <- lapply(row[, cat_names, drop = FALSE], function(x) as.integer(as.character(x) %||% 0))
    row[, cont_names] <- lapply(row[, cont_names, drop = FALSE], function(x) as.numeric(as.character(x) %||% 0))
    row
  })

  # Seed the 78 editable feature fields from the selected CSV row.
  # Missing / blank cells are filled with 0 and the user is told how many.
  sync_csv_to_inputs <- function() {
    if (is.null(input$csv_file)) return()
    df <- tryCatch(csv_data(), error = function(e) NULL)
    if (is.null(df) || nrow(df) == 0) return()
    n <- nrow(df)
    cur <- suppressWarnings(as.integer(input$cursor_index %||% 1))
    if (is.na(cur) || cur < 1 || cur > n) {
      if (!is.na(cur) && (cur < 1 || cur > n)) {
        showNotification(paste0("Current row must be between 1 and ", n, ". Adjusting."), type = "warning", duration = 4)
        updateNumericInput(session, "cursor_index", value = max(1L, min(cur, n)), max = n)
      }
      return()
    }
    row <- df[cur, feature_names, drop = FALSE]
    missing <- 0L
    vals <- setNames(vector("list", length(feature_names)), feature_names)
    for (feat in feature_names) {
      val <- row[[feat]]
      if (is.factor(val)) val <- as.character(val)
      num <- suppressWarnings(as.numeric(val))
      if (length(num) == 0 || is.na(num)) { num <- 0; missing <- missing + 1L }
      num <- if (feat %in% cat_names) as.integer(num) else as.numeric(num)
      vals[[feat]] <- num
      updateNumericInput(session, paste0("feat_", feat), value = num)
    }
    feature_vals(vals)  # write the source of truth synchronously (no client round-trip)
    if (missing > 0) {
      showNotification(sprintf("Row %d has %d missing value(s); filled with 0.", cur, missing),
                       type = "warning", duration = 5)
    }
  }

  # (csv load already calls sync_csv_to_inputs above; only re-sync on row change)
  observeEvent(input$cursor_index, sync_csv_to_inputs(), ignoreInit = TRUE)

  # Up/down buttons for row selection
  observeEvent(input$cursor_up, {
    cur <- suppressWarnings(as.integer(input$cursor_index %||% 1))
    if (is.na(cur) || cur <= 1) return()
    updateNumericInput(session, "cursor_index", value = cur - 1L)
  })
  observeEvent(input$cursor_down, {
    df <- tryCatch(csv_data(), error = function(e) NULL)
    if (is.null(df) || nrow(df) == 0) return()
    cur <- suppressWarnings(as.integer(input$cursor_index %||% 1))
    if (is.na(cur) || cur >= nrow(df)) return()
    updateNumericInput(session, "cursor_index", value = cur + 1L)
  })

  # Cursor change → stale both panels
  observeEvent(input$cursor_index, {
    rv$result <- NULL
    rv$cp_result <- NULL
    rv$importance <- NULL
  }, ignoreInit = TRUE)

  # Any feature edit → stale both panels
  observeEvent(manualData(), {
    rv$result <- NULL
    rv$cp_result <- NULL
    rv$importance <- NULL
  }, ignoreInit = TRUE)

  # Input df for prediction: the single row currently in the editable fields
  input_df <- reactive({
    df <- manualData()
    names(df) <- feature_names
    # Robust conversion: handle factor, character, NA
    for (f in cat_names) {
      v <- df[[f]]
      if (is.factor(v)) v <- as.character(v)
      df[[f]] <- as.integer(suppressWarnings(as.numeric(v)) %||% 0L)
    }
    for (f in cont_names) {
      v <- df[[f]]
      if (is.factor(v)) v <- as.character(v)
      df[[f]] <- as.numeric(suppressWarnings(as.numeric(v)) %||% 0)
    }
    df
  })

  rv <- reactiveValues(result = NULL, nrows = 0, importance = NULL, importance_version = 0L, cp_result = NULL)

  # ── CP upper bound: select outcome (AF / death) and current patient row ─────
  cp_current <- reactive({
    cp <- rv$cp_result
    if (is.null(cp)) return(NULL)
    ev  <- if (identical(input$risk_head, "AF")) cp$af else cp$death
    if (is.null(ev)) return(NULL)
    row <- tryCatch(display_row_idx(), error = function(e) 1L)
    n   <- length(ev$upper_bound_year)
    row <- max(1L, min(as.integer(row), n))
    list(
      ub_year   = round(ev$upper_bound_year[[row]], 2),
      ub_bin    = ev$upper_bound_bin[[row]],
      conf      = round(cp$target_coverage * 100),
      gamma     = cp$gamma,
      label     = if (identical(input$risk_head, "AF")) "AF" else "All-cause death"
    )
  })

  # ── CP upper bound UI renderer ─────────────────────────────────────────────
  output$cp_upper_bound_ui <- renderUI({
    cur <- cp_current()
    if (is.null(cur)) {
      # placeholder before first prediction
      div(
        style = "background:#f0f4ff; border-left:4px solid #4a90d9;
                 padding:10px 14px; border-radius:4px; margin-bottom:10px;",
        tags$small(style = "color:#666;",
          icon("info-circle"), " Run Prediction to see the conformal upper bound."
        )
      )
    } else {
      div(
        style = "background:#e8f5e9; border-left:4px solid #2e7d32;
                 padding:10px 14px; border-radius:4px; margin-bottom:10px;",
        tags$strong(style = "color:#1b5e20; font-size:1.05em;",
          icon("shield-alt"),
          sprintf(" %s Upper Bound: %.2f years", cur$label, cur$ub_year)
        ),
        tags$br(),
        tags$small(style = "color:#2e7d32;",
          sprintf("With %d%% confidence, %s is expected within %.2f years",
                  cur$conf, cur$label, cur$ub_year)
        ),
        tags$br(),
        tags$small(style = "color:#888; font-size:0.8em;",
          sprintf("(gamma = %.2f | AJ-recalibrated UPB)", cur$gamma)
        )
      )
    }
  })
  # ──────────────────────────────────────────────────────────────────────────

  # Reference data for permutation importance: sample replacement values from here
  reference_df <- reactive({
    ref <- NULL
    if (!is.null(input$csv_file)) {
      df <- tryCatch(csv_data(), error = function(e) NULL)
      if (!is.null(df) && nrow(df) > 1) {
        idx <- cursor_index_valid()
        # Exclude current row so we sample from other individuals
        if (!is.null(idx)) ref <- df[-idx, feature_names, drop = FALSE]
      }
    }
    # Single-row CSV (or none): fall back to example_input.csv for reference
    if (is.null(ref) || nrow(ref) == 0) {
      ref_paths <- c(
        "example_input.csv",
        file.path(getwd(), "example_input.csv"),
        example_path,
        file.path(getwd(), example_path)
      )
      for (path in unique(ref_paths)) {
        if (!is.null(path) && nzchar(path) && file.exists(path)) {
          tmp <- tryCatch(read.csv(path, check.names = FALSE, stringsAsFactors = FALSE), error = function(e) NULL)
          if (!is.null(tmp) && nrow(tmp) >= 1 && all(feature_names %in% names(tmp))) {
            ref <- tmp[, feature_names, drop = FALSE]
            rownames(ref) <- NULL
            break
          }
        }
      }
    }
    ref
  })

  observeEvent(input$predict, {
    rv$result <- NULL  # Clear previous result first to avoid stale/crash state
    rv$cp_result <- NULL
    err_msg <- NULL
    if (is.null(input$csv_file)) {
      showNotification("Please upload a CSV file first.", type = "error", duration = 5)
      return()
    }
    csv_now <- tryCatch(csv_data(), error = function(e) NULL)
    if (is.null(csv_now) || nrow(csv_now) < 1) {
      showNotification("CSV could not be read or is empty. Please re-upload.", type = "error", duration = 6)
      return()
    }
    if (is.null(cursor_index_valid())) {
      showNotification("Current row is out of range. Pick a valid row.", type = "error", duration = 6)
      return()
    }
    df <- tryCatch(
      input_df(),
      error = function(e) {
        err_msg <<- e$message
        NULL
      }
    )
    if (is.null(df) || nrow(df) < 1) {
      showNotification(
        if (nzchar(err_msg %||% "")) paste("Error loading data:", err_msg)
        else "No row selected or row out of range. Check Current row.",
        type = "error", duration = 8
      )
      return()
    }
    x_cat <- tryCatch(as.matrix(df[, cat_names, drop = FALSE]), error = function(e) NULL)
    x_cont <- tryCatch(as.matrix(df[, cont_names, drop = FALSE]), error = function(e) NULL)
    if (is.null(x_cat) || is.null(x_cont)) {
      showNotification("Error preparing data for prediction.", type = "error", duration = 5)
      return()
    }
    if (any(is.na(x_cat)) || any(is.na(x_cont))) {
      showNotification("Data contains NA values. Please check your CSV.", type = "error", duration = 5)
      return()
    }
    meta_path_abs <- tryCatch(normalizePath("model_meta.json", mustWork = TRUE), error = function(e) NULL)
    if (is.null(meta_path_abs)) {
      showNotification("model_meta.json not found.", type = "error", duration = 5)
      return()
    }

    row_info <- paste("row", input$cursor_index %||% 1)
    showNotification(paste("Running prediction for", row_info, "..."), type = "message", duration = 2)
    res <- tryCatch(
      {
        load_model(MODEL_PATH)  # Ensure model is loaded before each prediction
        message("Calling Python predict... rows: ", nrow(df))
        predict_from_arrays(
          x_cat = x_cat,
          x_cont = x_cont,
          model_path = MODEL_PATH,
          meta_path = meta_path_abs
        )
      },
      error = function(e) e
    )
    if (inherits(res, "error")) {
      rv$result <- NULL
      rv$cp_result <- NULL
      showNotification(paste("Prediction error:", res$message), type = "error", duration = 8)
      return()
    }
    rv$result <- res
    rv$nrows <- nrow(df)

    # ── CP upper bound (AF + all-cause death), AJ-recalibrated UPB ──────────
    recalib_path_abs <- normalizePath("cp_outputs/ucttrp_aj_upb_recalibration.npz", mustWork = FALSE)
    if (file.exists(recalib_path_abs)) {
      cp_res <- tryCatch(
        predict_event_upper_bounds(
          x_cat        = x_cat,
          x_cont       = x_cont,
          model_path   = MODEL_PATH,
          meta_path    = meta_path_abs,
          recalib_path = recalib_path_abs
        ),
        error = function(e) { message("CP error: ", e$message); NULL }
      )
      rv$cp_result <- cp_res
    } else {
      rv$cp_result <- NULL
    }
    # ──────────────────────────────────────────────────────────────────────

    showNotification("Prediction complete", type = "message", duration = 3)
  })

  # Sanitize data.frame for Python (ensure numeric types, no factors)
  sanitize_for_python <- function(df) {
    if (is.null(df) || nrow(df) == 0) return(NULL)
    out <- df[, feature_names, drop = FALSE]
    rownames(out) <- NULL
    # Fill NA per cell (not per column): a single NA must not zero the whole
    # column — that would corrupt the multi-row permutation reference.
    for (f in cat_names) {
      z <- suppressWarnings(as.numeric(as.character(out[[f]]))); z[is.na(z)] <- 0
      out[[f]] <- as.integer(z)
    }
    for (f in cont_names) {
      z <- suppressWarnings(as.numeric(as.character(out[[f]]))); z[is.na(z)] <- 0
      out[[f]] <- as.numeric(z)
    }
    out
  }

  # Compute permutation importance for the current sample (same row sent to prediction)
  observeEvent(input$compute_importance, {
    rv$importance <- NULL
    if (is.null(input$csv_file)) {
      showNotification("Please upload a CSV file first.", type = "error", duration = 5)
      return()
    }
    csv_now <- tryCatch(csv_data(), error = function(e) NULL)
    if (is.null(csv_now) || nrow(csv_now) < 1) {
      showNotification("CSV could not be read or is empty. Please re-upload.", type = "error", duration = 6)
      return()
    }
    if (is.null(cursor_index_valid())) {
      showNotification("Current row is out of range. Pick a valid row.", type = "error", duration = 6)
      return()
    }
    test_row <- tryCatch(sanitize_for_python(cursor_row()), error = function(e) NULL)
    ref <- tryCatch(sanitize_for_python(reference_df()), error = function(e) NULL)
    if (is.null(test_row) || nrow(test_row) == 0) {
      showNotification("No sample to analyze. Select a valid row.", type = "error", duration = 5)
      return()
    }
    if (is.null(ref) || nrow(ref) < 1) {
      showNotification("No reference data for permutation. Upload a multi-row CSV, or add example_input.csv for reference.", type = "error", duration = 6)
      return()
    }
    meta_path_abs <- tryCatch(
      normalizePath("model_meta.json", mustWork = TRUE),
      error = function(e) {
        p <- file.path(getwd(), "model_meta.json")
        if (file.exists(p)) normalizePath(p) else NULL
      }
    )
    if (is.null(meta_path_abs)) {
      showNotification("model_meta.json not found. Run app from ShinyApp directory.", type = "error", duration = 5)
      return()
    }
    risk_idx <- match(input$risk_head, risk_names)
    if (is.na(risk_idx)) risk_idx <- 1L
    risk_idx <- risk_idx - 1L
    time_bin <- suppressWarnings(as.integer(input$importance_time_bin %||% "-2"))
    if (is.na(time_bin)) time_bin <- -2L

    # Importance is always computed over all 78 features
    features_to_permute <- as.list(feature_names)
    non_top_defaults_py <- list()

    showNotification(
      paste0("Computing feature importance (", length(feature_names), " passes, all features)..."),
      type = "message", duration = 3
    )
    imp <- tryCatch(
      permutation_importance_from_dataframe(
        df_test_row = test_row,
        df_ref = ref,
        model_path = MODEL_PATH,
        meta_path = meta_path_abs,
        risk_idx = risk_idx,
        time_bin = time_bin,
        features_to_permute = features_to_permute,
        non_top_defaults = non_top_defaults_py
      ),
      error = function(e) e
    )
    if (inherits(imp, "error")) {
      showNotification(paste("Importance error:", imp$message), type = "error", duration = 10)
      return()
    }
    rv$importance <- imp
    rv$importance_version <- rv$importance_version + 1L
    showNotification("Feature importance complete", type = "message", duration = 3)
  })

  # Display row index: clamp to prediction result rows; depends on cursor so outputs refresh on cursor change
  display_row_idx <- reactive({
    req(rv$result)
    num_rows <- dim(rv$result$prob)[1]
    idx <- as.integer(input$cursor_index %||% 1)
    max(1, min(idx, num_rows))
  })

  # Current selected series/table values (both event prob and survival)
  current_prediction_df <- reactive({
    req(rv$result)
    res <- rv$result
    probs <- res$prob
    surv <- res$surv
    num_risks <- dim(probs)[2]
    num_bins <- dim(probs)[3]

    row_idx <- display_row_idx()
    risk_idx <- max(1, min(match(input$risk_head, res$risk_names), num_risks))
    duration <- if (length(res$duration_labels) == num_bins) res$duration_labels else seq_len(num_bins)

    prob_vals <- as.numeric(probs[row_idx, risk_idx, ])
    surv_vals <- as.numeric(surv[row_idx, risk_idx, ])
    list(
      data = data.frame(
        Duration = duration,
        Event_probability = prob_vals,
        Survival = surv_vals,
        stringsAsFactors = FALSE
      ),
      risk_name = res$risk_names[risk_idx],
      row_idx = row_idx
    )
  })

  # Helper: x-position (bin index) of the CP upper bound for the current plot
  cp_vline_x <- reactive({
    cur <- cp_current()
    if (is.null(cur)) return(NULL)
    list(
      x       = cur$ub_bin + 1L,   # R is 1-indexed; Python bin 0 → idx 1
      ub_year = cur$ub_year,
      conf    = cur$conf
    )
  })

  # Shared helper: x-axis breaks + labels. Show at most ~10 evenly spaced ticks
  # so labels stay legible even when the model emits many bins (e.g. 30).
  .xaxis <- function(n_bins, mode) {
    step   <- max(1L, ceiling(n_bins / 10))
    breaks <- seq(1L, n_bins, by = step)
    denom  <- max(1L, n_bins - 1L)
    if (isTRUE(mode == "year")) {
      list(
        breaks = breaks,
        labels = round((breaks - 1L) / denom * 15.0, 1),
        title  = "Follow-up (years)"
      )
    } else {
      list(breaks = breaks, labels = as.character(breaks), title = "Bin")
    }
  }

  prediction_plot_prob <- reactive({
    cur    <- current_prediction_df()
    df_plot <- cur$data
    df_plot$idx <- seq_len(nrow(df_plot))
    n_bins <- nrow(df_plot)
    xa     <- .xaxis(n_bins, input$xaxis_mode)
    vl     <- cp_vline_x()

    p <- ggplot(df_plot, aes(x = idx, y = Event_probability)) +
      geom_line(color = "steelblue") +
      geom_point(color = "steelblue") +
      scale_x_continuous(breaks = xa$breaks, labels = xa$labels) +
      labs(title = paste(cur$risk_name, "- Event probability"),
           x = xa$title, y = "Event probability") +
      theme_minimal() +
      theme(axis.text.x = element_text(size = 7))

    if (!is.null(vl)) {
      p <- p +
        geom_vline(xintercept = vl$x, color = "red", linewidth = 0.9, linetype = "dashed") +
        annotate("text",
                 x     = vl$x + 0.25,
                 y     = max(df_plot$Event_probability) * 0.95,
                 label = sprintf("%d%% UB: %.2f yr", vl$conf, vl$ub_year),
                 color = "red", size = 3, hjust = 0)
    }
    p
  })

  prediction_plot_surv <- reactive({
    cur    <- current_prediction_df()
    df_plot <- cur$data
    df_plot$idx <- seq_len(nrow(df_plot))
    n_bins <- nrow(df_plot)
    xa     <- .xaxis(n_bins, input$xaxis_mode)
    vl     <- cp_vline_x()

    p <- ggplot(df_plot, aes(x = idx, y = Survival)) +
      geom_line(color = "steelblue") +
      geom_point(color = "steelblue") +
      scale_x_continuous(breaks = xa$breaks, labels = xa$labels) +
      labs(title = paste(cur$risk_name, "- Survival"),
           x = xa$title, y = "Survival") +
      theme_minimal() +
      theme(axis.text.x = element_text(size = 7))

    if (!is.null(vl)) {
      p <- p +
        geom_vline(xintercept = vl$x, color = "red", linewidth = 0.9, linetype = "dashed") +
        annotate("text",
                 x     = vl$x + 0.25,
                 y     = 0.97,
                 label = sprintf("%d%% UB: %.2f yr", vl$conf, vl$ub_year),
                 color = "red", size = 3, hjust = 0)
    }
    p
  })

  output$predictionTable <- renderDT({
    if (is.null(rv$result)) {
      return(datatable(
        data.frame(Message = "Click 'Run Prediction' to see event probability and survival by bin."),
        rownames = FALSE,
        options = list(dom = "t", paging = FALSE, searching = FALSE, ordering = FALSE)
      ))
    }
    df_out <- current_prediction_df()$data
    df_out$Event_probability <- round(df_out$Event_probability, 4)
    df_out$Survival <- round(df_out$Survival, 4)
    datatable(df_out, rownames = FALSE, options = list(dom = "t", pageLength = 10))
  })

  output$predictionPlot_prob <- renderPlot({
    if (is.null(rv$result)) {
      par(mar = c(2, 2, 2, 2))
      plot.new()
      text(0.5, 0.5, "Click 'Run Prediction' to see plot.", cex = 1.2)
      return(invisible(NULL))
    }
    prediction_plot_prob()
  })

  output$predictionPlot_surv <- renderPlot({
    if (is.null(rv$result)) {
      par(mar = c(2, 2, 2, 2))
      plot.new()
      text(0.5, 0.5, "Click 'Run Prediction' to see plot.", cex = 1.2)
      return(invisible(NULL))
    }
    prediction_plot_surv()
  })

  output$download_table_csv <- downloadHandler(
    filename = function() {
      req(rv$result)
      cur <- current_prediction_df()
      risk <- gsub("[^A-Za-z0-9]+", "_", tolower(cur$risk_name))
      paste0("prediction_", risk, "_row_", input$cursor_index %||% 1, ".csv")
    },
    content = function(file) {
      req(rv$result)
      df_out <- current_prediction_df()$data
      df_out$Event_probability <- round(df_out$Event_probability, 4)
      df_out$Survival <- round(df_out$Survival, 4)
      write.csv(df_out, file, row.names = FALSE)
    }
  )

  output$download_plot_png <- downloadHandler(
    filename = function() {
      req(rv$result)
      cur <- current_prediction_df()
      risk <- gsub("[^A-Za-z0-9]+", "_", tolower(cur$risk_name))
      paste0("prediction_curves_", risk, "_row_", input$cursor_index %||% 1, ".png")
    },
    content = function(file) {
      req(rv$result)
      p_prob <- prediction_plot_prob()
      p_surv <- prediction_plot_surv()
      png(file, width = 8, height = 10, units = "in", res = 300)
      gridExtra::grid.arrange(p_prob, p_surv, ncol = 1)
      dev.off()
    }
  )

  # Clear importance when new prediction runs (user must re-click Compute Importance)
  observeEvent(rv$result, {
    rv$importance <- NULL
  }, ignoreNULL = FALSE, ignoreInit = TRUE)

  # Feature importance outputs
  importance_df <- reactive({
    rv$importance_version  # explicit dependency — forces invalidation on every new computation
    if (is.null(rv$importance)) return(NULL)
    imp <- rv$importance
    # Robustly convert Python dict / reticulate object to named R numeric vector
    if (!is.list(imp) && !is.numeric(imp)) {
      imp <- tryCatch(reticulate::py_to_r(imp), error = function(e) NULL)
    }
    if (is.null(imp)) return(NULL)
    nms <- names(imp)
    vals <- tryCatch(
      vapply(imp, function(x) {
        v <- suppressWarnings(as.numeric(x))
        if (length(v) == 0 || is.na(v[1])) 0 else v[1]
      }, numeric(1)),
      error = function(e) NULL
    )
    if (is.null(vals) || is.null(nms) || length(nms) != length(vals)) return(NULL)
    df <- data.frame(Feature = nms, Importance = vals, stringsAsFactors = FALSE)
    df <- df[order(-df$Importance, df$Feature), , drop = FALSE]
    df
  })

  # Scope note for Feature Importance panel
  output$importance_scope_note <- renderUI({
    helpText("Scope: all 78 features from the current record.",
             style = "font-size:0.82em; color:#666; font-style:italic;")
  })

  output$importancePlot <- renderPlot({    df <- importance_df()
    if (is.null(df) || nrow(df) == 0) {
      par(mar = c(2, 2, 2, 2))
      plot.new()
      text(0.5, 0.5, "Click 'Compute Importance' to see which features\nmost influence this individual's prediction.", cex = 0.95)
      return(invisible(NULL))
    }
    df_nonzero <- df[df$Importance > 0, , drop = FALSE]
    if (nrow(df_nonzero) == 0) {
      par(mar = c(2, 2, 2, 2))
      plot.new()
      text(0.5, 0.5, "All feature importances are zero.\nThe prediction may be insensitive to feature changes\nfor this patient with the current reference data.", cex = 0.9)
      return(invisible(NULL))
    }
    top_n <- min(20, nrow(df_nonzero))
    top <- head(df_nonzero, top_n)

    # Normalize to % of total sensitivity
    total <- sum(df_nonzero$Importance)
    top$Pct <- top$Importance / total * 100

    # Shared y-axis order (highest importance at top)
    feat_levels <- rev(top$Feature)
    top$Feature <- factor(top$Feature, levels = feat_levels)

    # Left plot: % of total
    p_pct <- ggplot(top, aes(x = Pct, y = Feature)) +
      geom_col(fill = "steelblue", width = 0.7) +
      geom_text(aes(label = sprintf("%.1f%%", Pct)), hjust = -0.1, size = 2.8) +
      scale_x_continuous(expand = expansion(mult = c(0, 0.18))) +
      labs(x = "% of total sensitivity", y = NULL,
           title = "Relative importance") +
      theme_minimal() +
      theme(axis.text.y = element_text(size = 8),
            axis.text.x = element_text(size = 7),
            plot.title = element_text(size = 9, face = "bold"))

    # Right plot: raw absolute change (scientific notation)
    p_raw <- ggplot(top, aes(x = Importance, y = Feature)) +
      geom_col(fill = "coral3", width = 0.7) +
      scale_x_continuous(labels = scales::scientific,
                         expand = expansion(mult = c(0, 0.15))) +
      labs(x = "|Δ prediction|", y = NULL,
           title = "Raw sensitivity") +
      theme_minimal() +
      theme(axis.text.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.x = element_text(size = 7),
            plot.title = element_text(size = 9, face = "bold"))

    gridExtra::grid.arrange(p_pct, p_raw, ncol = 2, widths = c(1.6, 1))
  })

  output$importanceTable <- renderDT({
    df <- importance_df()
    if (is.null(df) || nrow(df) == 0) {
      return(datatable(
        data.frame(Message = "Click 'Compute Importance' to see feature importance for the current sample."),
        rownames = FALSE,
        options = list(dom = "t", paging = FALSE, searching = FALSE, ordering = FALSE)
      ))
    }
    total <- sum(df$Importance[df$Importance > 0])
    df$Pct <- ifelse(total > 0, round(df$Importance / total * 100, 2), 0)
    df$Importance <- formatC(df$Importance, format = "e", digits = 3)
    names(df) <- c("Feature", "Raw |Δ prediction|", "% of total")
    datatable(df, rownames = FALSE, options = list(pageLength = 15, order = list(list(2, "desc"))))
  })

  output$download_importance_csv <- downloadHandler(
    filename = function() {
      risk <- gsub("[^A-Za-z0-9]+", "_", tolower(input$risk_head %||% "AF"))
      paste0("feature_importance_", risk, ".csv")
    },
    content = function(file) {
      df <- importance_df()
      if (is.null(df) || nrow(df) == 0) {
        write.csv(data.frame(Message = "No importance data. Compute first."), file, row.names = FALSE)
        return()
      }
      write.csv(df, file, row.names = FALSE)
    }
  )
}

shinyApp(ui = ui, server = server)
