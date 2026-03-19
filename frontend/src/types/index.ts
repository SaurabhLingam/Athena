export type DataMode = "labeled" | "unlabeled";
export type Profile = "standard" | "finance" | "healthcare" | "nlp" | "custom";
export type AppView = "landing" | "upload" | "report" | "compare";

export interface AnalysisResult {
  mode: DataMode;
  eda: EDAResult;
  ml_diagnostics: MLDiagnostics;
  timeseries?: TimeSeriesResult;
  model_arena?: ModelArenaResult;
}

export interface TimeSeriesResult {
  detected: boolean;
  reason?: string;
  datetime_col?: string;
  target_col?: string;
  n_observations?: number;
  freq?: string;
  eda?: TimeSeriesEDA;
  baselines?: TimeSeriesBaselines;
}

export interface TimeSeriesEDA {
  summary: {
    mean: number;
    std: number;
    min: number;
    max: number;
    median: number;
    missing_pct: number;
  };
  stationarity: {
    adf: StationarityTest | null;
    kpss: StationarityTest | null;
    verdict: "stationary" | "non_stationary" | "inconclusive" | "unknown";
    needs_differencing: boolean;
  };
  decomposition: {
    period: number;
    trend: (number | null)[];
    seasonal: (number | null)[];
    residual: (number | null)[];
    dates: string[];
    error?: string;
  };
  acf_pacf: {
    acf: number[];
    pacf: number[];
    confidence: number;
    lags: number[];
    error?: string;
  };
  missing_timestamps: {
    gap_count: number;
    total_expected: number;
    coverage_pct: number;
    gaps: string[];
    error?: string;
  };
  rolling_stats: {
    window: number;
    mean: (number | null)[];
    std: (number | null)[];
    dates: string[];
    values: number[];
    error?: string;
  };
}

export interface StationarityTest {
  statistic: number;
  p_value: number;
  is_stationary: boolean;
  critical_values: Record<string, number>;
  interpretation: string;
  error?: string;
}

export interface TimeSeriesBaseline {
  rmse?: number;
  mae?: number;
  mape?: number;
  model?: string;
  forecast?: number[];
  order?: number[];
  window?: number;
  error?: string;
}

export interface TimeSeriesBaselines {
  naive?: TimeSeriesBaseline;
  moving_average?: TimeSeriesBaseline;
  exp_smoothing?: TimeSeriesBaseline;
  arima?: TimeSeriesBaseline;
  ranking?: { model: string; rmse: number; mae: number; mape: number }[];
  error?: string;
}

export interface TSDetectionResult {
  datetime_candidates: string[];
  target_candidates: string[];
  auto_datetime: string | null;
  auto_target: string | null;
}

export interface EDAResult {
  basic: BasicInfo;
  col_views: ColumnViews;
  missingness: Missingness;
  numeric_summary: NumericSummary[];
  categorical_summary: CategoricalSummary[];
  outliers: OutlierResult;
  similar_pairs: SimilarPairs;
  feature_redundancy: FeatureRedundancy;
  data_structural_report: DataStructuralReport;
  free_text?: FreeTextResult;
  feature_correlation?: FeatureCorrelation;
  target_insights?: TargetInsights;
  bivariate?: BivariateResult;
  nlp?: NLPResult;
}

export interface BasicInfo {
  rows: number;
  cols: number;
  col_names: string[];
  unique_values: Record<string, number>;
}

export interface ColumnViews {
  target: string[];
  identifiers: string[];
  entity_keys: string[];
  numeric_features: string[];
  categorical_features: string[];
}

export interface Missingness {
  missing_values_percent_cols: Record<string, number>;
  flagged_columns: string[];
  row_missingness_flags: Record<string, number>;
  severely_incomplete_rows: number[];
  row_missing_distribution: Record<string, number>;
  systematic_missingness: any[];
}

export interface NumericSummary {
  column: string;
  count: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  Q1: number;
  Q3: number;
  skewness: number;
  zero_pct: number;
  iqr: number;
  missing_pct: number;
}

export interface CategoricalSummary {
  column: string;
  n_unique: number;
  top_categories: Record<string, number>;
  missing_pct: number;
  most_frequent_pct: number;
  high_cardinality: boolean;
}

export interface OutlierResult {
  column_outliers: ColumnOutlier[];
  dataset_outlier_diagnosis: {
    heavy_tailed_features: string[];
    safe_features: string[];
    outliers_are_signal: boolean;
  };
}

export interface ColumnOutlier {
  column: string;
  skewness: number;
  kurtosis: number;
  outlier_pct: number;
  distribution_type: string;
  suggest_log_transform: boolean;
  suggest_robust_loss: boolean;
}

export interface SimilarPairs {
  highly_similar_features: any[];
  summary: { potential_multicollinearity: boolean };
}

export interface FeatureRedundancy {
  highly_correlated_pairs: any[];
  overall_redundancy_risk: string;
}

export interface DataStructuralReport {
  critical_alerts: string[];
  action_plan: {
    drop_columns: string[];
    apply_log_transform: string[];
    apply_log1p_transform: string[];
  };
  feature_redundancy: any[];
  categorical_health: any[];
  data_structural_score: number;
}

export interface FreeTextResult {
  free_text_columns: { column: string; avg_length: number; unique_ratio: number; recommendation: string }[];
  count: number;
}

export interface FeatureCorrelation {
  numeric_target_correlation: Record<string, { pearson: number; spearman: number }>;
  categorical_target_impact: { column: string; distribution_shift?: number; likely_influential: boolean }[];
}

export interface TargetInsights {
  target_name: string;
  missing_pct: number;
  task_type: "classification" | "regression";
  class_distribution?: Record<string, number>;
  imbalance_ratio?: number;
  is_imbalanced?: boolean;
  numeric_feature_separation?: Record<string, any>;
  target_stats?: Record<string, number>;
  likely_transform_needed?: boolean;
}

export interface BivariateResult {
  bivariate_results: any[];
  target_col: string;
  task_type: string;
}

export interface MLDiagnostics {
  status: string;
  task_type?: string;
  metric?: string;
  ML_readiness_score?: number;
  component_scores?: Record<string, number>;
  penalties?: Record<string, string[]>;
  recommendations?: Record<string, string[]>;
  baseline?: BaselineResult;
  leakage?: LeakageResult;
  stability?: StabilityResult;
  clustering?: any;
  anomaly_detection?: any;
}

export interface BaselineResult {
  mean_score: number;
  std_score: number;
  fold_scores: number[];
  fold_curves?: {
    fold: number;
    train: number[];
    val: number[];
  }[];
  metric: string;
  n_folds: number;
  learning_dynamics: {
    mean_best_iteration: number;
    mean_train_val_gap: number;
    overfitting_detected: boolean;
    underfitting_detected: boolean;
    mean_improvement: number;
  };
}

export interface LeakageResult {
  leakage_score: number;
  suspicious_features: string[];
}

export interface StabilityResult {
  mean_score: number;
  std_score: number;
  is_unstable: boolean;
  loss_gap_ratio: number;
}

export interface DriftResult {
  overall_drift_score: number;
  column_drift: ColumnDrift[];
  new_categories: Record<string, string[]>;
  missing_rate_delta: Record<string, number>;
  summary: string;
}

export interface ColumnDrift {
  column: string;
  drift_score: number;
  train_mean?: number;
  test_mean?: number;
  ks_statistic?: number;
  ks_pvalue?: number;
}

export interface NLPWordEntry {
  word: string;
  count: number;
}

export interface NLPColumnResult {
  column: string;
  text_stats: {
    avg_word_count: number;
    avg_char_length: number;
    max_word_count: number;
    min_word_count: number;
    vocab_size: number;
    lexical_diversity: number;
    top_words: NLPWordEntry[];
  };
  text_quality: {
    empty_ratio: number;
    duplicate_ratio: number;
    url_ratio: number;
    number_ratio: number;
    special_char_ratio: number;
  };
  ml_readiness: {
    vocab_size: number;
    vectorization_recommendation: string;
    suitable_for_embeddings: boolean;
  };
  per_class_analysis?: {
    class: string;
    count: number;
    avg_word_count: number;
    avg_char_length: number;
    top_words: NLPWordEntry[];
  }[];
}

export interface NLPResult {
  nlp_analysis: NLPColumnResult[];
  text_col_count: number;
}

export interface ModelArenaResult {
  status: string;
  reason?: string;
  task_type?: string;
  metric?: string;
  n_folds?: number;
  n_features?: number;
  n_samples?: number;
  winner?: string;
  models?: ModelArenaModel[];
  ranking?: ModelArenaRanking[];
}
 
export interface ModelArenaModel {
  model: string;
  score: number | null;
  rmse: number | null;
  std: number | null;
  time_s: number | null;
  error: string | null;
}
 
export interface ModelArenaRanking {
  rank: number;
  model: string;
  score: number | null;
  rmse: number | null;
  std: number | null;
  time_s: number | null;
}