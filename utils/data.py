import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import os

@st.cache_data
def load_superstore(uploaded_file=None):
    """Load and normalize superstore data with caching"""
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
        
        # Normalize column names: lowercase, replace non-alphanumerics with _, collapse whitespace
        df.columns = df.columns.str.lower().str.replace(r'[^a-z0-9]+', '_', regex=True).str.strip('_')
        
        # Alias common variants to canonical names
        column_aliases = {
            'order_date': 'order_date',
            'orderdate': 'order_date',
            'date': 'order_date',
            'sales': 'sales',
            'revenue': 'sales',
            'profit': 'profit',
            'customer_id': 'customer_id',
            'customerid': 'customer_id',
            'cust_id': 'customer_id',
            'customer_name': 'customer_name',
            'customername': 'customer_name',
            'cust_name': 'customer_name',
            'product_name': 'product_name',
            'productname': 'product_name',
            'product': 'product_name',
            'category': 'category',
            'sub_category': 'sub_category',
            'subcategory': 'sub_category',
            'region': 'region',
            'state': 'state',
            'city': 'city',
            'discount': 'discount',
            'quantity': 'quantity',
            'qty': 'quantity',
            'ship_mode': 'ship_mode',
            'shipmode': 'ship_mode',
            'shipping_mode': 'ship_mode',
            'segment': 'segment',
            'order_id': 'order_id',
            'orderid': 'order_id',
            'shipping_cost': 'shipping_cost',
            'shippingcost': 'shipping_cost',
            'ship_cost': 'shipping_cost'
        }
        
        # Apply aliases
        df = df.rename(columns=column_aliases)
        
        # Parse order_date to datetime
        if 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        
        # Coerce numerics
        numeric_columns = ['sales', 'profit', 'discount', 'quantity', 'shipping_cost']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Strip whitespace on key categoricals
        categorical_columns = ['category', 'sub_category', 'region', 'state', 'city', 'segment', 'ship_mode', 'product_name', 'customer_name']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Remove rows with missing critical data
        critical_columns = ['order_date', 'sales', 'profit']
        for col in critical_columns:
            if col in df.columns:
                df = df.dropna(subset=[col])
        
        # Ensure required columns exist
        required_columns = ['customer_id', 'order_date', 'sales', 'profit', 'category', 'region']
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
        'categories': sorted(df['category'].unique().tolist()) if 'category' in df.columns else [],
        'regions': sorted(df['region'].unique().tolist()) if 'region' in df.columns else [],
        'segments': sorted(df['segment'].unique().tolist()) if 'segment' in df.columns else []
    }

def apply_global_filters(df, categories, regions):
    """Apply global filters to dataframe"""
    if not categories or not regions:
        return df.iloc[0:0]  # empty safe DF
    
    out = df.copy()
    if 'category' in out:
        out = out[out['category'].isin(categories)]
    if 'region' in out:
        out = out[out['region'].isin(regions)]
    
    return out

def guard_empty(d):
    """Guard against empty dataframe and show user message"""
    if d.empty:
        st.info("No results — relax filters (Category/Region).")
        st.stop()

def get_date_range_options(df):
    """Get predefined date range options"""
    if df.empty or 'order_date' not in df.columns:
        return {}
    
    min_date = df['order_date'].min().date()
    max_date = df['order_date'].max().date()
    
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
    critical_columns = ['customer_id', 'order_date', 'sales', 'profit']
    for col in critical_columns:
        if col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 5:
                warnings.append(f"{col} has {missing_pct:.1f}% missing values")
    
    # Check for suspicious data patterns
    if 'profit' in df.columns:
        negative_profit_pct = (df['profit'] < 0).sum() / len(df) * 100
        if negative_profit_pct > 30:
            warnings.append(f"High percentage of negative profit orders: {negative_profit_pct:.1f}%")
    
    if 'discount' in df.columns:
        high_discount_pct = (df['discount'] > 0.5).sum() / len(df) * 100
        if high_discount_pct > 10:
            warnings.append(f"High percentage of orders with >50% discount: {high_discount_pct:.1f}%")
    
    return warnings