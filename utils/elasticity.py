import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

@st.cache_data
def analyze_discount_elasticity(df):
    """Analyze discount elasticity using binning approach for performance"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Create discount bands for analysis
        df_copy = df.copy()
        
        # Define discount bins
        discount_bins = [0, 0.1, 0.2, 0.3, 0.4, 1.0]
        discount_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40%+']
        
        df_copy['Discount_Band'] = pd.cut(
            df_copy['Discount'],
            bins=discount_bins,
            labels=discount_labels,
            include_lowest=True
        )
        
        # Calculate metrics by discount band (using median for performance)
        elasticity_data = df_copy.groupby('Discount_Band').agg({
            'Sales': ['median', 'sum', 'count'],
            'Profit': ['median', 'sum'],
            'Order.ID': 'nunique',
            'Discount': 'median'
        }).reset_index()
        
        # Flatten column names
        elasticity_data.columns = [
            'Discount_Band', 'Median_Sales_Per_Order', 'Total_Sales', 'Record_Count',
            'Median_Profit_Per_Order', 'Total_Profit', 'Order_Count', 'Median_Discount'
        ]
        
        # Calculate additional metrics
        elasticity_data['Profit_Margin'] = (
            elasticity_data['Total_Profit'] / elasticity_data['Total_Sales']
        ) * 100
        
        elasticity_data['Avg_Discount'] = elasticity_data['Median_Discount']
        
        # Create discount band midpoints for plotting
        band_midpoints = {
            '0-10%': 0.05,
            '10-20%': 0.15,
            '20-30%': 0.25,
            '30-40%': 0.35,
            '40%+': 0.45
        }
        
        elasticity_data['Discount_Band_Mid'] = elasticity_data['Discount_Band'].map(band_midpoints)
        
        # Calculate elasticity coefficients
        if len(elasticity_data) > 2:
            try:
                # Price elasticity of profit
                profit_slope, _, profit_r, _, _ = stats.linregress(
                    elasticity_data['Discount_Band_Mid'], 
                    elasticity_data['Median_Profit_Per_Order']
                )
                
                # Price elasticity of sales volume  
                sales_slope, _, sales_r, _, _ = stats.linregress(
                    elasticity_data['Discount_Band_Mid'], 
                    elasticity_data['Median_Sales_Per_Order']
                )
                
                elasticity_data['Profit_Elasticity_Coef'] = profit_slope
                elasticity_data['Sales_Elasticity_Coef'] = sales_slope
                elasticity_data['Profit_R_Squared'] = profit_r ** 2
                elasticity_data['Sales_R_Squared'] = sales_r ** 2
                
            except Exception:
                # Fallback if regression fails
                elasticity_data['Profit_Elasticity_Coef'] = 0
                elasticity_data['Sales_Elasticity_Coef'] = 0
                elasticity_data['Profit_R_Squared'] = 0
                elasticity_data['Sales_R_Squared'] = 0
        
        return elasticity_data
        
    except Exception as e:
        st.error(f"Error analyzing discount elasticity: {str(e)}")
        return pd.DataFrame()

def simulate_profit_impact(df, max_discount_rate):
    """Simulate profit impact of capping discounts at specified rate"""
    if df.empty:
        return {'current_profit': 0, 'simulated_profit': 0}
    
    try:
        # Calculate current total profit
        current_profit = df['Profit'].sum()
        
        # Create simulation dataframe
        sim_df = df.copy()
        
        # Identify orders with discounts above the cap
        high_discount_mask = sim_df['Discount'] > max_discount_rate
        
        if high_discount_mask.sum() == 0:
            # No orders exceed the discount cap
            return {
                'current_profit': current_profit,
                'simulated_profit': current_profit
            }
        
        # For orders exceeding the discount cap, estimate new profit
        # Assumption: reducing discount proportionally increases profit
        for idx in sim_df[high_discount_mask].index:
            original_discount = sim_df.loc[idx, 'Discount']
            original_sales = sim_df.loc[idx, 'Sales']
            original_profit = sim_df.loc[idx, 'Profit']
            
            # Calculate original cost (Sales - Profit)
            original_cost = original_sales - original_profit
            
            # Calculate what sales would be with reduced discount
            # Assuming same base price, new sales = cost / (1 - new_discount_rate)
            base_price = original_sales / (1 - original_discount)
            new_sales = base_price * (1 - max_discount_rate)
            new_profit = new_sales - original_cost
            
            sim_df.loc[idx, 'Simulated_Profit'] = new_profit
        
        # For orders not exceeding the cap, keep original profit
        sim_df.loc[~high_discount_mask, 'Simulated_Profit'] = sim_df.loc[~high_discount_mask, 'Profit']
        
        simulated_profit = sim_df['Simulated_Profit'].sum()
        
        return {
            'current_profit': current_profit,
            'simulated_profit': simulated_profit,
            'orders_affected': high_discount_mask.sum(),
            'avg_current_discount': df[high_discount_mask]['Discount'].mean() if high_discount_mask.sum() > 0 else 0
        }
        
    except Exception as e:
        st.error(f"Error simulating profit impact: {str(e)}")
        return {'current_profit': 0, 'simulated_profit': 0}

@st.cache_data
def get_optimal_discount_recommendations(df):
    """Get discount optimization recommendations by product category"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Analyze by sub-category
        category_analysis = df.groupby('Sub.Category').agg({
            'Sales': 'sum',
            'Profit': 'sum', 
            'Discount': ['mean', 'std'],
            'Order.ID': 'nunique'
        }).reset_index()
        
        # Flatten column names
        category_analysis.columns = [
            'Sub_Category', 'Total_Sales', 'Total_Profit',
            'Avg_Discount', 'Discount_Std', 'Order_Count'
        ]
        
        # Calculate profit margin
        category_analysis['Profit_Margin'] = (
            category_analysis['Total_Profit'] / category_analysis['Total_Sales']
        ) * 100
        
        # Filter for categories with sufficient data
        category_analysis = category_analysis[category_analysis['Order_Count'] >= 10]
        
        # Identify optimization opportunities
        # Low margin + High discount = Reduce discount opportunity  
        category_analysis['Reduce_Discount_Opportunity'] = (
            (category_analysis['Profit_Margin'] < 10) & 
            (category_analysis['Avg_Discount'] > 0.2)
        )
        
        # High margin + Low discount = Increase sales volume opportunity
        category_analysis['Increase_Volume_Opportunity'] = (
            (category_analysis['Profit_Margin'] > 20) & 
            (category_analysis['Avg_Discount'] < 0.1)
        )
        
        # Calculate recommended discount ranges
        def get_discount_recommendation(row):
            if row['Reduce_Discount_Opportunity']:
                # Recommend reducing discount by 50%
                return max(0, row['Avg_Discount'] * 0.5)
            elif row['Increase_Volume_Opportunity']:
                # Recommend small increase in discount for volume
                return min(0.3, row['Avg_Discount'] + 0.05)
            else:
                return row['Avg_Discount']
        
        category_analysis['Recommended_Discount'] = category_analysis.apply(get_discount_recommendation, axis=1)
        
        # Sort by potential impact (high sales, large discount change)
        category_analysis['Discount_Change'] = abs(
            category_analysis['Recommended_Discount'] - category_analysis['Avg_Discount']
        )
        category_analysis['Impact_Score'] = (
            category_analysis['Total_Sales'] * category_analysis['Discount_Change']
        )
        
        return category_analysis.sort_values('Impact_Score', ascending=False)
        
    except Exception as e:
        st.error(f"Error getting discount recommendations: {str(e)}")
        return pd.DataFrame()

def create_elasticity_charts(elasticity_data):
    """Create interactive elasticity visualization charts"""
    if elasticity_data.empty:
        return None, None
    
    try:
        # Profit elasticity chart
        profit_fig = px.scatter(
            elasticity_data,
            x='Discount_Band_Mid',
            y='Median_Profit_Per_Order',
            size='Order_Count',
            hover_data=['Discount_Band', 'Total_Profit', 'Profit_Margin'],
            title="Discount vs Profit Elasticity",
            labels={
                'Discount_Band_Mid': 'Discount Rate',
                'Median_Profit_Per_Order': 'Median Profit per Order ($)'
            }
        )
        
        # Add trend line
        if len(elasticity_data) > 1:
            z = np.polyfit(elasticity_data['Discount_Band_Mid'], elasticity_data['Median_Profit_Per_Order'], 1)
            p = np.poly1d(z)
            profit_fig.add_trace(
                go.Scatter(
                    x=elasticity_data['Discount_Band_Mid'],
                    y=p(elasticity_data['Discount_Band_Mid']),
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', dash='dash')
                )
            )
        
        profit_fig.update_traces(marker=dict(color='darkred', opacity=0.7))
        profit_fig.update_layout(showlegend=False)
        
        # Sales elasticity chart
        sales_fig = px.scatter(
            elasticity_data,
            x='Discount_Band_Mid',
            y='Median_Sales_Per_Order',
            size='Order_Count',
            hover_data=['Discount_Band', 'Total_Sales'],
            title="Discount vs Sales Volume Elasticity",
            labels={
                'Discount_Band_Mid': 'Discount Rate',
                'Median_Sales_Per_Order': 'Median Sales per Order ($)'
            }
        )
        
        # Add trend line
        if len(elasticity_data) > 1:
            z = np.polyfit(elasticity_data['Discount_Band_Mid'], elasticity_data['Median_Sales_Per_Order'], 1)
            p = np.poly1d(z)
            sales_fig.add_trace(
                go.Scatter(
                    x=elasticity_data['Discount_Band_Mid'],
                    y=p(elasticity_data['Discount_Band_Mid']),
                    mode='lines',
                    name='Trend',
                    line=dict(color='blue', dash='dash')
                )
            )
        
        sales_fig.update_traces(marker=dict(color='darkblue', opacity=0.7))
        sales_fig.update_layout(showlegend=False)
        
        return profit_fig, sales_fig
        
    except Exception as e:
        st.error(f"Error creating elasticity charts: {str(e)}")
        return None, None
