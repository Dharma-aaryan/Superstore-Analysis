import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from operator import attrgetter

@st.cache_data
def calculate_rfm(df):
    """Calculate RFM (Recency, Frequency, Monetary) scores for customers"""
    if df.empty or 'Customer.ID' not in df.columns:
        return pd.DataFrame()
    
    try:
        # Set analysis date (latest order date + 1 day)
        analysis_date = df['Order.Date'].max() + pd.Timedelta(days=1)
        
        # Calculate RFM metrics per customer
        rfm_data = df.groupby('Customer.ID').agg({
            'Order.Date': lambda x: (analysis_date - x.max()).days,  # Recency
            'Order.ID': 'nunique',  # Frequency  
            'Sales': 'sum'  # Monetary
        }).reset_index()
        
        rfm_data.columns = ['Customer.ID', 'Recency', 'Frequency', 'Monetary']
        
        # Remove customers with zero or negative monetary value
        rfm_data = rfm_data[rfm_data['Monetary'] > 0]
        
        if rfm_data.empty:
            return pd.DataFrame()
        
        # Calculate RFM scores using quartiles (1-5 scale)
        # For Recency: lower is better (more recent)
        rfm_data['R_Score'] = pd.qcut(
            rfm_data['Recency'], 
            q=5, 
            labels=[5, 4, 3, 2, 1],  # Reverse order for recency
            duplicates='drop'
        ).astype(str).astype(int)
        
        # For Frequency: higher is better
        rfm_data['F_Score'] = pd.qcut(
            rfm_data['Frequency'].rank(method='first'), 
            q=5, 
            labels=[1, 2, 3, 4, 5],
            duplicates='drop'
        ).astype(str).astype(int)
        
        # For Monetary: higher is better
        rfm_data['M_Score'] = pd.qcut(
            rfm_data['Monetary'].rank(method='first'), 
            q=5, 
            labels=[1, 2, 3, 4, 5],
            duplicates='drop'
        ).astype(str).astype(int)
        
        # Calculate combined RFM score
        rfm_data['RFM_Score'] = (
            rfm_data['R_Score'].astype(str) + 
            rfm_data['F_Score'].astype(str) + 
            rfm_data['M_Score'].astype(str)
        )
        
        return rfm_data
        
    except Exception as e:
        st.error(f"Error calculating RFM: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def segment_customers(rfm_data):
    """Segment customers into named tiers based on RFM scores"""
    if rfm_data.empty:
        return pd.DataFrame()
    
    try:
        rfm_segmented = rfm_data.copy()
        
        # Define segmentation rules
        def categorize_customer(row):
            r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
            
            # VIP: High value, frequent, recent customers
            if r >= 4 and f >= 4 and m >= 4:
                return 'VIP'
            
            # Loyal: Good frequency and monetary, may not be very recent
            elif f >= 4 and m >= 3:
                return 'Loyal'
            
            # Promising: Recent customers with potential
            elif r >= 4 and (f >= 2 or m >= 2):
                return 'Promising'
            
            # At-Risk: All others (low recency, frequency, or monetary)
            else:
                return 'At-Risk'
        
        rfm_segmented['Tier'] = rfm_segmented.apply(categorize_customer, axis=1)
        
        return rfm_segmented
        
    except Exception as e:
        st.error(f"Error segmenting customers: {str(e)}")
        return pd.DataFrame()

def get_rfm_insights(customer_segments):
    """Generate business insights from RFM segmentation"""
    if customer_segments.empty:
        return []
    
    insights = []
    
    try:
        # Overall distribution
        tier_counts = customer_segments['Tier'].value_counts()
        total_customers = len(customer_segments)
        
        # VIP insights
        if 'VIP' in tier_counts.index:
            vip_count = tier_counts['VIP']
            vip_percentage = (vip_count / total_customers) * 100
            vip_revenue = customer_segments[customer_segments['Tier'] == 'VIP']['Monetary'].sum()
            total_revenue = customer_segments['Monetary'].sum()
            vip_revenue_share = (vip_revenue / total_revenue) * 100 if total_revenue > 0 else 0
            
            insights.append(
                f"**{vip_count} VIP customers** ({vip_percentage:.1f}%) generate **{vip_revenue_share:.1f}%** of total revenue"
            )
        
        # At-Risk insights
        if 'At-Risk' in tier_counts.index:
            atrisk_count = tier_counts['At-Risk']
            atrisk_percentage = (atrisk_count / total_customers) * 100
            
            # Calculate average recency for at-risk customers
            atrisk_avg_recency = customer_segments[
                customer_segments['Tier'] == 'At-Risk'
            ]['Recency'].mean()
            
            insights.append(
                f"**{atrisk_count} At-Risk customers** ({atrisk_percentage:.1f}%) haven't ordered in **{atrisk_avg_recency:.0f} days** on average"
            )
        
        # Loyal customer insights
        if 'Loyal' in tier_counts.index:
            loyal_avg_frequency = customer_segments[
                customer_segments['Tier'] == 'Loyal'
            ]['Frequency'].mean()
            
            insights.append(
                f"**Loyal customers** average **{loyal_avg_frequency:.1f} orders** each, representing consistent business value"
            )
        
        # Promising insights
        if 'Promising' in tier_counts.index:
            promising_count = tier_counts['Promising']
            promising_avg_monetary = customer_segments[
                customer_segments['Tier'] == 'Promising'
            ]['Monetary'].mean()
            
            insights.append(
                f"**{promising_count} Promising customers** show growth potential with **${promising_avg_monetary:,.0f}** average spending"
            )
        
        return insights[:4]  # Return top 4 insights
        
    except Exception as e:
        st.error(f"Error generating RFM insights: {str(e)}")
        return []

@st.cache_data  
def get_customer_lifetime_value(df, customer_segments):
    """Calculate estimated customer lifetime value by segment"""
    if df.empty or customer_segments.empty:
        return pd.DataFrame()
    
    try:
        # Calculate additional metrics for CLV estimation
        customer_metrics = df.groupby('Customer.ID').agg({
            'Order.Date': ['min', 'max'],
            'Sales': ['sum', 'mean'],
            'Profit': 'sum',
            'Order.ID': 'nunique'
        }).reset_index()
        
        # Flatten column names
        customer_metrics.columns = ['Customer.ID', 'First_Order', 'Last_Order', 
                                   'Total_Sales', 'Avg_Order_Value', 'Total_Profit', 'Order_Count']
        
        # Calculate customer lifespan in days
        customer_metrics['Lifespan_Days'] = (
            customer_metrics['Last_Order'] - customer_metrics['First_Order']
        ).dt.days
        
        # Merge with RFM segments
        clv_data = customer_segments.merge(customer_metrics, on='Customer.ID', how='left')
        
        # Calculate estimated CLV metrics by tier
        clv_summary = clv_data.groupby('Tier').agg({
            'Total_Sales': 'mean',
            'Avg_Order_Value': 'mean',
            'Order_Count': 'mean',
            'Lifespan_Days': 'mean',
            'Total_Profit': 'mean'
        }).round(2)
        
        # Estimate annual CLV (simple projection)
        clv_summary['Est_Annual_CLV'] = (
            clv_summary['Avg_Order_Value'] * 
            (clv_summary['Order_Count'] / (clv_summary['Lifespan_Days'] / 365 + 1))
        ).fillna(clv_summary['Total_Sales'])
        
        return clv_summary.reset_index()
        
    except Exception as e:
        st.error(f"Error calculating CLV: {str(e)}")
        return pd.DataFrame()

def get_retention_analysis(df):
    """Analyze customer retention patterns"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Create cohort analysis based on first order month
        df_copy = df.copy()
        df_copy['Order_Period'] = df_copy['Order.Date'].dt.to_period('M')
        df_copy['First_Order_Period'] = df_copy.groupby('Customer.ID')['Order.Date'].transform('min').dt.to_period('M')
        
        # Calculate period number (months since first order)
        df_copy['Period_Number'] = (df_copy['Order_Period'] - df_copy['First_Order_Period']).apply(attrgetter('n'))
        
        # Create cohort table
        cohort_data = df_copy.groupby(['First_Order_Period', 'Period_Number'])['Customer.ID'].nunique().reset_index()
        cohort_table = cohort_data.pivot(index='First_Order_Period', columns='Period_Number', values='Customer.ID')
        
        # Calculate cohort sizes (customers in each cohort)
        cohort_sizes = df_copy.groupby('First_Order_Period')['Customer.ID'].nunique()
        
        # Calculate retention rates
        retention_table = cohort_table.divide(cohort_sizes, axis=0)
        
        return retention_table
        
    except Exception as e:
        st.error(f"Error calculating retention analysis: {str(e)}")
        return pd.DataFrame()
