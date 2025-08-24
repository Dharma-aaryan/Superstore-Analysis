import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import os

@st.cache_data
def load_and_clean_data(uploaded_file=None):
    """Load and clean superstore data with caching"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            # Fallback to default superstore.csv
            if os.path.exists('superstore.csv'):
                df = pd.read_csv('superstore.csv')
            else:
                st.error("Default superstore.csv file not found. Please upload a CSV file.")
                return pd.DataFrame()
        
        if df.empty:
            return pd.DataFrame()
        
        # Data cleaning pipeline
        df = df.copy()
        
        # Parse dates
        if 'Order.Date' in df.columns:
            df['Order.Date'] = pd.to_datetime(df['Order.Date'], errors='coerce')
        
        if 'Ship.Date' in df.columns:
            df['Ship.Date'] = pd.to_datetime(df['Ship.Date'], errors='coerce')
        
        # Clean numeric columns
        numeric_columns = ['Sales', 'Profit', 'Quantity', 'Discount', 'Shipping.Cost']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Trim whitespace from string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            if col not in ['Order.Date', 'Ship.Date']:
                df[col] = df[col].astype(str).str.strip()
        
        # Remove rows with missing critical data
        critical_columns = ['Order.Date', 'Sales', 'Profit']
        for col in critical_columns:
            if col in df.columns:
                df = df.dropna(subset=[col])
        
        # Ensure required columns exist
        required_columns = ['Customer.ID', 'Order.Date', 'Sales', 'Profit', 'Category', 'Region']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def get_filter_values(df):
    """Get unique values for filter dropdowns"""
    if df.empty:
        return {'categories': [], 'regions': [], 'segments': []}
    
    return {
        'categories': sorted(df['Category'].unique().tolist()) if 'Category' in df.columns else [],
        'regions': sorted(df['Region'].unique().tolist()) if 'Region' in df.columns else [],
        'segments': sorted(df['Segment'].unique().tolist()) if 'Segment' in df.columns else []
    }

def apply_global_filters(df, date_range, categories, regions):
    """Apply global filters to dataframe"""
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Date range filter
    if len(date_range) == 2 and date_range[0] and date_range[1]:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        filtered_df = filtered_df[
            (filtered_df['Order.Date'] >= start_date) & 
            (filtered_df['Order.Date'] <= end_date)
        ]
    
    # Category filter
    if categories and 'Category' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    
    # Region filter
    if regions and 'Region' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    return filtered_df

def get_date_range_options(df):
    """Get predefined date range options"""
    if df.empty or 'Order.Date' not in df.columns:
        return {}
    
    min_date = df['Order.Date'].min().date()
    max_date = df['Order.Date'].max().date()
    
    return {
        'All Time': (min_date, max_date),
        'Last 365 Days': (max_date - pd.Timedelta(days=365), max_date),
        'Last 180 Days': (max_date - pd.Timedelta(days=180), max_date),
        'Last 90 Days': (max_date - pd.Timedelta(days=90), max_date),
        'Current Year': (date(max_date.year, 1, 1), max_date)
    }

def validate_data_quality(df):
    """Validate data quality and return warnings"""
    warnings = []
    
    if df.empty:
        warnings.append("Dataset is empty")
        return warnings
    
    # Check for missing values in critical columns
    critical_columns = ['Customer.ID', 'Order.Date', 'Sales', 'Profit']
    for col in critical_columns:
        if col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 5:
                warnings.append(f"{col} has {missing_pct:.1f}% missing values")
    
    # Check for suspicious data patterns
    if 'Profit' in df.columns:
        negative_profit_pct = (df['Profit'] < 0).sum() / len(df) * 100
        if negative_profit_pct > 30:
            warnings.append(f"High percentage of negative profit orders: {negative_profit_pct:.1f}%")
    
    if 'Discount' in df.columns:
        high_discount_pct = (df['Discount'] > 0.5).sum() / len(df) * 100
        if high_discount_pct > 10:
            warnings.append(f"High percentage of orders with >50% discount: {high_discount_pct:.1f}%")
    
    return warnings
