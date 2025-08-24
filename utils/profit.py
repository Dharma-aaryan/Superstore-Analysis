import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def analyze_profitability_blackholes(df, top_n=20, groupby_level='product_name'):
    """Analyze profit black holes - high sales but negative profit items"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Group by specified level (product_name or sub_category)
        if groupby_level not in df.columns:
            groupby_level = 'sub_category' if 'sub_category' in df.columns else 'product_name'
        
        # Calculate aggregated metrics by item
        profit_analysis = df.groupby(groupby_level).agg({
            'sales': 'sum',
            'profit': 'sum',
            'quantity': 'sum',
            'order_id': 'nunique'
        }).reset_index()
        
        profit_analysis.columns = ['Item', 'Total_Sales', 'Total_Profit', 'Total_Quantity', 'Order_Count']
        
        # Filter for loss-making items with significant sales
        blackholes = profit_analysis[
            (profit_analysis['Total_Profit'] < 0) & 
            (profit_analysis['Total_Sales'] > profit_analysis['Total_Sales'].median())
        ].copy()
        
        if blackholes.empty:
            return pd.DataFrame()
        
        # Calculate additional metrics
        blackholes['Loss'] = blackholes['Total_Profit']  # Negative values
        blackholes['Loss_Per_Sale'] = blackholes['Loss'] / blackholes['Total_Sales']
        blackholes['Avg_Sales_Per_Order'] = blackholes['Total_Sales'] / blackholes['Order_Count']
        
        # Sort by total loss (most negative first)
        blackholes = blackholes.sort_values('Loss').head(top_n)
        
        return blackholes[['Item', 'Total_Sales', 'Loss', 'Loss_Per_Sale', 'Order_Count', 'Avg_Sales_Per_Order']]
        
    except Exception as e:
        st.error(f"Error analyzing profit blackholes: {str(e)}")
        return pd.DataFrame()

def get_pareto_chart(blackholes_df):
    """Create Pareto chart for loss analysis"""
    if blackholes_df.empty:
        return None
    
    try:
        # Prepare data for Pareto chart
        df_sorted = blackholes_df.copy()
        df_sorted['Abs_Loss'] = abs(df_sorted['Loss'])
        df_sorted = df_sorted.sort_values('Abs_Loss', ascending=False)
        
        # Calculate cumulative percentage
        df_sorted['Cumulative_Loss'] = df_sorted['Abs_Loss'].cumsum()
        df_sorted['Cumulative_Pct'] = (df_sorted['Cumulative_Loss'] / df_sorted['Abs_Loss'].sum()) * 100
        
        # Create Pareto chart
        fig = go.Figure()
        
        # Bar chart for losses
        fig.add_trace(go.Bar(
            x=df_sorted['Item'],
            y=df_sorted['Abs_Loss'],
            name='Loss Amount',
            marker_color='red',
            opacity=0.7
        ))
        
        # Line chart for cumulative percentage
        fig.add_trace(go.Scatter(
            x=df_sorted['Item'],
            y=df_sorted['Cumulative_Pct'],
            mode='lines+markers',
            name='Cumulative %',
            yaxis='y2',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        # Add 80% reference line
        fig.add_hline(y=80, line_dash="dash", line_color="green", 
                     annotation_text="80% Threshold", yref='y2')
        
        # Update layout
        fig.update_layout(
            title="Pareto Analysis: Loss Drivers (20/80 Rule)",
            xaxis_title="Items",
            yaxis=dict(
                title="Loss Amount ($)",
                side="left",
                showgrid=True
            ),
            yaxis2=dict(
                title="Cumulative Percentage (%)",
                side="right",
                overlaying="y",
                range=[0, 100],
                showgrid=False
            ),
            legend=dict(x=0.7, y=0.9),
            height=500
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating Pareto chart: {str(e)}")
        return None

@st.cache_data
def get_loss_drivers_by_segment(df):
    """Analyze loss drivers by customer segment and category"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Group by segment and category
        segment_analysis = df.groupby(['Segment', 'Category', 'Region']).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Discount': 'mean',
            'Order.ID': 'nunique'
        }).reset_index()
        
        # Calculate profit margin
        segment_analysis['Profit_Margin'] = (
            segment_analysis['Profit'] / segment_analysis['Sales']
        ) * 100
        
        # Filter for problematic combinations
        problems = segment_analysis[
            (segment_analysis['Profit'] < 0) | 
            (segment_analysis['Profit_Margin'] < -5)
        ].copy()
        
        # Sort by profit (worst first)
        problems = problems.sort_values('Profit')
        
        return problems
        
    except Exception as e:
        st.error(f"Error analyzing loss drivers by segment: {str(e)}")
        return pd.DataFrame()

def get_profit_improvement_opportunities(df):
    """Identify specific opportunities for profit improvement"""
    if df.empty:
        return []
    
    opportunities = []
    
    try:
        # High discount, low profit items
        high_discount_items = df[
            (df['Discount'] > 0.3) & 
            (df['Profit'] < 0)
        ].groupby('Product.Name').agg({
            'Profit': 'sum',
            'Sales': 'sum',
            'Discount': 'mean'
        }).sort_values('Profit').head(5)
        
        for item, data in high_discount_items.iterrows():
            opportunities.append({
                'type': 'Discount Optimization',
                'item': item,
                'current_loss': abs(data['Profit']),
                'avg_discount': data['Discount'],
                'recommendation': f"Reduce discount on {item} from {data['Discount']:.1%} to improve ${abs(data['Profit']):,.0f} loss"
            })
        
        # Low-margin, high-volume categories
        category_margins = df.groupby('Category').agg({
            'Profit': 'sum',
            'Sales': 'sum',
            'Order.ID': 'nunique'
        })
        category_margins['Margin'] = (category_margins['Profit'] / category_margins['Sales']) * 100
        
        low_margin_categories = category_margins[
            (category_margins['Margin'] < 5) & 
            (category_margins['Order.ID'] > 100)
        ].sort_values('Margin')
        
        for category, data in low_margin_categories.iterrows():
            opportunities.append({
                'type': 'Category Review',
                'item': category,
                'current_margin': data['Margin'],
                'order_volume': data['Order.ID'],
                'recommendation': f"Review {category} pricing strategy - {data['Margin']:.1f}% margin on {data['Order.ID']} orders"
            })
        
        return opportunities[:10]  # Return top 10 opportunities
        
    except Exception as e:
        st.error(f"Error identifying improvement opportunities: {str(e)}")
        return []

def calculate_discount_impact(df):
    """Calculate the impact of different discount levels on profitability"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Create discount bands
        df_copy = df.copy()
        df_copy['Discount_Band'] = pd.cut(
            df_copy['Discount'],
            bins=[0, 0.1, 0.2, 0.3, 0.4, 1.0],
            labels=['0-10%', '10-20%', '20-30%', '30-40%', '40%+'],
            include_lowest=True
        )
        
        # Calculate metrics by discount band
        discount_impact = df_copy.groupby('Discount_Band').agg({
            'Sales': ['sum', 'mean'],
            'Profit': ['sum', 'mean'],
            'Order.ID': 'nunique',
            'Quantity': 'mean'
        }).round(2)
        
        # Flatten column names
        discount_impact.columns = ['_'.join(col).strip() for col in discount_impact.columns.values]
        discount_impact = discount_impact.reset_index()
        
        # Calculate additional metrics
        discount_impact['Profit_Margin'] = (
            discount_impact['Profit_sum'] / discount_impact['Sales_sum']
        ) * 100
        
        discount_impact['Revenue_Per_Order'] = (
            discount_impact['Sales_sum'] / discount_impact['Order.ID_nunique']
        )
        
        return discount_impact
        
    except Exception as e:
        st.error(f"Error calculating discount impact: {str(e)}")
        return pd.DataFrame()
