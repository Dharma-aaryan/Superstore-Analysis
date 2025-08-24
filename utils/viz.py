import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# US state full-name → 2-letter mapping dict
US_STATE_MAPPING = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'District of Columbia': 'DC'
}

@st.cache_data
def create_kpi_cards(df):
    """Calculate KPI metrics for executive summary"""
    if df.empty:
        return {
            'total_sales': 0,
            'total_profit': 0,
            'profit_margin': 0,
            'total_orders': 0,
            'avg_discount': 0,
            'total_customers': 0
        }
    
    try:
        metrics = {
            'total_sales': df['sales'].sum() if 'sales' in df.columns else 0,
            'total_profit': df['profit'].sum() if 'profit' in df.columns else 0,
            'profit_margin': (df['profit'].sum() / df['sales'].sum()) if 'profit' in df.columns and 'sales' in df.columns and df['sales'].sum() > 0 else 0,
            'total_orders': df['order_id'].nunique() if 'order_id' in df.columns else len(df),
            'avg_discount': df['discount'].mean() if 'discount' in df.columns else 0,
            'total_customers': df['customer_id'].nunique() if 'customer_id' in df.columns else 0
        }
        
        return metrics
        
    except Exception as e:
        st.error(f"Error calculating KPIs: {str(e)}")
        return {
            'total_sales': 0,
            'total_profit': 0, 
            'profit_margin': 0,
            'total_orders': 0,
            'avg_discount': 0,
            'total_customers': 0
        }

def create_radar_chart(blackhole_data):
    """Create radar chart for top profitability black holes"""
    if blackhole_data.empty:
        return None
    
    try:
        # Prepare data for radar chart
        top_5 = blackhole_data.head(5).copy()
        
        # Normalize metrics for radar display
        max_loss = abs(top_5['Loss'].min())  # Most negative loss
        max_sales = top_5['Total_Sales'].max()
        
        # Create normalized metrics (0-100 scale)
        radar_data = []
        categories = []
        
        for _, row in top_5.iterrows():
            loss_normalized = (abs(row['Loss']) / max_loss) * 100
            sales_normalized = (row['Total_Sales'] / max_sales) * 100
            
            radar_data.append([loss_normalized, sales_normalized, row['Order_Count']])
            categories.append(row['Item'][:30])  # Truncate long names
        
        # Create radar chart
        fig = go.Figure()
        
        for i, (item, values) in enumerate(zip(categories, radar_data)):
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=['Loss Impact', 'Sales Volume', 'Order Count', 'Loss Impact'],
                fill='toself',
                name=item,
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Top 5 Profitability Black Holes - Multi-dimensional Impact",
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating radar chart: {str(e)}")
        return None

@st.cache_data
def plot_state_map(df, value_col='profit', title='Profit by State'):
    """Create US state choropleth map that respects filters"""
    if df.empty or 'state' not in df.columns:
        return None
    
    try:
        # Group the filtered dataframe by state and aggregate
        state_data = df.groupby('state').agg({
            'profit': 'sum',
            'sales': 'sum', 
            'discount': 'mean'
        }).reset_index()
        
        # Map state to 2-letter state_code; drop rows with unknown codes
        state_data['state_code'] = state_data['state'].map(US_STATE_MAPPING)
        state_data = state_data.dropna(subset=['state_code'])
        
        if state_data.empty:
            return None
        
        # Create choropleth map
        fig = px.choropleth(
            state_data,
            locations='state_code',
            color=value_col,
            hover_data={
                'sales': ':$,.0f',
                'discount': ':.1%'
            },
            locationmode='USA-states',
            color_continuous_scale='RdYlGn',
            scope='usa',
            title=title
        )
        
        fig.update_layout(height=600)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating choropleth map: {str(e)}")
        return None

def create_segment_heatmap(data, x_col, y_col, value_col, title):
    """Create a generic heatmap for segment analysis"""
    if data.empty:
        return None
    
    try:
        # Pivot data for heatmap
        heatmap_data = data.pivot_table(
            index=y_col,
            columns=x_col, 
            values=value_col,
            aggfunc='mean'
        ).fillna(0)
        
        # Create heatmap
        fig = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale='RdYlGn',
            title=title,
            aspect='auto'
        )
        
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating heatmap: {str(e)}")
        return None

def create_profitability_waterfall(df):
    """Create waterfall chart showing profit drivers"""
    if df.empty:
        return None
    
    try:
        # Calculate profit impact by major categories
        categories = ['Technology', 'Furniture', 'Office Supplies']
        profits = []
        
        for cat in categories:
            if cat in df['Category'].values:
                cat_profit = df[df['Category'] == cat]['Profit'].sum()
                profits.append(cat_profit)
            else:
                profits.append(0)
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Profit Analysis",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=categories + ["Total"],
            textposition="outside",
            text=[f"${p:,.0f}" for p in profits] + [f"${sum(profits):,.0f}"],
            y=profits + [sum(profits)],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title="Profit Contribution by Category",
            showlegend=False,
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating waterfall chart: {str(e)}")
        return None

def create_trend_analysis(df, date_col='Order.Date', metric_col='Profit', freq='M'):
    """Create trend analysis chart"""
    if df.empty or date_col not in df.columns:
        return None
    
    try:
        # Resample data by specified frequency
        df_trend = df.set_index(date_col).resample(freq)[metric_col].sum().reset_index()
        
        # Create trend chart
        fig = px.line(
            df_trend,
            x=date_col,
            y=metric_col,
            title=f"{metric_col} Trend Over Time",
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=metric_col,
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating trend analysis: {str(e)}")
        return None

def create_distribution_chart(df, column, title=None):
    """Create distribution chart for numerical columns"""
    if df.empty or column not in df.columns:
        return None
    
    try:
        if title is None:
            title = f"Distribution of {column}"
        
        fig = px.histogram(
            df,
            x=column,
            nbins=30,
            title=title,
            opacity=0.7
        )
        
        # Add mean line
        mean_val = df[column].mean()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_val:.2f}")
        
        fig.update_layout(height=400)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating distribution chart: {str(e)}")
        return None

def create_correlation_matrix(df, columns=None):
    """Create correlation matrix heatmap"""
    if df.empty:
        return None
    
    try:
        if columns is None:
            # Select numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = numeric_cols[:10]  # Limit to first 10 numeric columns
        
        # Calculate correlation matrix
        corr_matrix = df[columns].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            color_continuous_scale='RdBu',
            title="Feature Correlation Matrix",
            zmin=-1,
            zmax=1
        )
        
        fig.update_layout(height=500)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating correlation matrix: {str(e)}")
        return None

def format_currency(value):
    """Format currency values for display"""
    if abs(value) >= 1e6:
        return f"${value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.0f}K"
    else:
        return f"${value:.0f}"

def format_percentage(value):
    """Format percentage values for display"""
    return f"{value:.1%}"

def format_large_number(value):
    """Format large numbers for display"""
    if abs(value) >= 1e6:
        return f"{value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.0f}K"
    else:
        return f"{value:.0f}"
