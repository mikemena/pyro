import pandas as pd

def analyze_dataset(file_path, sheet_name=None):
    """
    Analyze dataset structure and provide preprocessing recommendations
    Args:
        file_path: Path to the Excel file
        sheet_name: Specific sheet to analyze (optional)
    Returns:
        dict: Analysis results with column types and recommendations
    """
    # Load data
    if sheet_name is None:
        df = pd.read_excel(file_path, sheet_name=0) # First sheet
    else:
        df = pd.read_excel(file_path, sheet_name=sheet_name)

    print("Dataset Overview")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    analysis_results = {
        'shape': df.shape,
        'columns': {},
        'recommendations': []
    }

    # Analyze each column
    for col in df.columns:
        col_analysis = analyze_column(df[col], col)
        analysis_results['columns'][col] = col_analysis
        print(f"{col}")
        print(f"  Type: {col_analysis['recommended_type']}")
        print(f"  Unique values: {col_analysis['unique_count']}")
        print(f"  Missing: {col_analysis['missing_count']}")
        if col_analysis['sample_values']:
            print(f"  Sample: {col_analysis['sample_values']}")

        # Display recommendations for this column
        if col_analysis['recommendations']:
            print(f"  Recommendations: {'; '.join(col_analysis['recommendations'])}")

        # Add column recommendations to overall recommendations
        analysis_results['recommendations'].extend([f"{col}: {rec}" for rec in col_analysis['recommendations']])
        print()

    # Display overall recommendations
    if analysis_results['recommendations']:
        print("=== PREPROCESSING RECOMMENDATIONS ===")
        for i, rec in enumerate(analysis_results['recommendations'], 1):
            print(f"{i}. {rec}")
        print()

    return analysis_results

def analyze_column(series, col_name):
    """Analyze individual column and determine preprocessing needs"""
    # Get basic statistics
    stats = get_column_statistics(series)

    # Determine column type
    recommended_type = determine_column_type(series, stats['unique_count'])

    # Generate recommendations
    recommendations = generate_recommendations(series, recommended_type, stats)

    return {
        'unique_count': stats['unique_count'],
        'missing_count': stats['missing_count'],
        'missing_percentage': stats['missing_percentage'],
        'sample_values': stats['sample_values'],
        'recommended_type': recommended_type,
        'is_numeric': stats['is_numeric'],
        'is_datetime': stats['is_datetime'],
        'recommendations': recommendations
    }

def get_column_statistics(series):
    """Extract basic statistics from a pandas series"""
    unique_values = series.dropna().unique()
    unique_count = len(unique_values)
    missing_count = series.isnull().sum()
    total_count = len(series)
    missing_percentage = (missing_count / total_count) * 100

    # Sample values (first 5, converted to string)
    sample_values = [str(val) for val in unique_values[:5]]

    # Determine data types
    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)

    return {
        'unique_count': unique_count,
        'missing_count': missing_count,
        'missing_percentage': missing_percentage,
        'sample_values': sample_values,
        'is_numeric': is_numeric,
        'is_datetime': is_datetime,
        'total_count': total_count
    }

def determine_column_type(series, unique_count):
    """Determine the recommended column type based on data characteristics"""
    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)

    if is_datetime:
        return 'datetime'
    elif unique_count == 2 and not is_numeric:
        return 'binary'
    elif unique_count <= 10 and not is_numeric and not is_datetime:
        return 'categorical'
    elif is_numeric:
        return 'numerical'
    else:
        return 'text'

def generate_recommendations(series, recommended_type, stats):
    """Generate preprocessing recommendations based on column analysis"""
    recommendations = []

    # Add missing value recommendations
    recommendations.extend(get_missing_value_recommendations(stats, recommended_type))

    # Add type-specific recommendations
    recommendations.extend(get_type_specific_recommendations(series, recommended_type, stats))

    # Add data quality recommendations
    recommendations.extend(get_data_quality_recommendations(stats, recommended_type))

    return recommendations

def get_missing_value_recommendations(stats, recommended_type):
    """Generate recommendations for handling missing values"""
    recommendations = []
    missing_count = stats['missing_count']
    missing_percentage = stats['missing_percentage']

    if missing_count > 0:
        if missing_percentage > 50:
            recommendations.append(f"Consider dropping column (>{missing_percentage:.1f}% missing)")
        elif recommended_type == 'numerical':
            recommendations.append("Fill missing values with mean/median")
        elif recommended_type in ['categorical', 'binary']:
            recommendations.append("Fill missing values with mode or 'Unknown' category")
        elif recommended_type == 'datetime':
            recommendations.append("Fill missing dates with forward/backward fill or interpolation")
        else:
            recommendations.append("Fill missing text values with 'Unknown' or empty string")

    return recommendations

def get_type_specific_recommendations(series, recommended_type, stats):
    """Generate type-specific preprocessing recommendations"""
    recommendations = []
    unique_count = stats['unique_count']

    if recommended_type == 'categorical':
        recommendations.append("Consider one-hot encoding or label encoding")
        if unique_count > 5:
            recommendations.append("High cardinality - consider grouping rare categories")

    elif recommended_type == 'binary':
        recommendations.append("Consider binary encoding (0/1) or keep as categorical")

    elif recommended_type == 'numerical':
        recommendations.extend(get_numerical_recommendations(series))

    elif recommended_type == 'datetime':
        recommendations.extend(get_datetime_recommendations())

    elif recommended_type == 'text':
        recommendations.extend(get_text_recommendations(unique_count))

    return recommendations

def get_numerical_recommendations(series):
    """Generate recommendations for numerical columns"""
    recommendations = []

    # Check for outliers using IQR method
    outlier_count = detect_outliers(series)
    if outlier_count > 0:
        recommendations.append(f"Check for outliers ({outlier_count} potential outliers detected)")

    recommendations.append("Consider scaling/normalization for ML models")
    return recommendations

def get_datetime_recommendations():
    """Generate recommendations for datetime columns"""
    return [
        "Extract features: year, month, day, weekday, etc.",
        "Consider time-based features if relevant"
    ]

def get_text_recommendations(unique_count):
    """Generate recommendations for text columns"""
    recommendations = []
    if unique_count > 100:
        recommendations.append("High cardinality text - consider text preprocessing or feature extraction")
    recommendations.append("Consider text cleaning, tokenization, or encoding")
    return recommendations

def get_data_quality_recommendations(stats, recommended_type):
    """Generate data quality recommendations"""
    recommendations = []
    unique_count = stats['unique_count']
    total_count = stats['total_count']

    if unique_count == 1:
        recommendations.append("Column has only one unique value - consider dropping")

    if unique_count == total_count and recommended_type != 'categorical':
        recommendations.append("All values are unique - might be an ID column")

    return recommendations

def detect_outliers(series):
    """Detect outliers using IQR method"""
    if not pd.api.types.is_numeric_dtype(series):
        return 0

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return len(outliers)

if __name__ == "__main__":
    results = analyze_dataset('data/cred_data.xlsx')
