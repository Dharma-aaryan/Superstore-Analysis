import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(layout="wide", page_title="Superstore Insights")

@st.cache_data
def load_data(path="superstore.csv"):
    """Load and return the superstore dataset"""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Dataset not found at {path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def normalize_columns(df):
    """Normalize column names to standard format"""
    if df.empty:
        return df
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Column mapping for various naming conventions
    column_mapping = {
        # Order Date variations
        'order date': 'order_date',
        'order.date': 'order_date',
        'orderdate': 'order_date',
        'date': 'order_date',
        
        # Sales variations
        'sales': 'sales',
        'revenue': 'sales',
        'gmv': 'sales',
        
        # Profit variations
        'profit': 'profit',
        
        # Discount variations
        'discount': 'discount',
        
        # Order ID variations
        'order id': 'order_id',
        'order.id': 'order_id',
        'orderid': 'order_id',
        'order_id': 'order_id',
        
        # Customer variations
        'customer id': 'customer_id',
        'customer.id': 'customer_id',
        'customerid': 'customer_id',
        'customer_id': 'customer_id',
        'customer name': 'customer_name',
        'customer.name': 'customer_name',
        'customername': 'customer_name',
        'customer_name': 'customer_name',
        
        # Category variations
        'category': 'category',
        'sub category': 'sub_category',
        'sub.category': 'sub_category',
        'subcategory': 'sub_category',
        'sub_category': 'sub_category',
        
        # Geography variations
        'region': 'region',
        'state': 'state',
        'city': 'city',
        
        # Shipping variations
        'ship mode': 'ship_mode',
        'ship.mode': 'ship_mode',
        'shipmode': 'ship_mode',
        'ship_mode': 'ship_mode',
        
        # Other variations
        'segment': 'segment',
        'quantity': 'quantity'
    }
    
    # Normalize column names: lowercase and clean
    df.columns = df.columns.str.lower().str.strip()
    
    # Apply mapping
    df = df.rename(columns=column_mapping)
    
    # Convert data types
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    
    numeric_cols = ['sales', 'profit', 'discount', 'quantity']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.dropna(subset=['sales', 'profit'] if all(col in df.columns for col in ['sales', 'profit']) else [])

def compute_kpis(df):
    """Compute executive KPIs"""
    if df.empty:
        return {'gmv': 0, 'gross_profit': 0, 'margin_pct': 0, 'orders': 0}
    
    gmv = df['sales'].sum() if 'sales' in df.columns else 0
    gross_profit = df['profit'].sum() if 'profit' in df.columns else 0
    margin_pct = (gross_profit / gmv * 100) if gmv > 0 else 0
    orders = df['order_id'].nunique() if 'order_id' in df.columns else len(df)
    
    return {
        'gmv': gmv,
        'gross_profit': gross_profit,
        'margin_pct': margin_pct,
        'orders': orders
    }

def format_currency(value):
    """Format currency values in compact notation"""
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.2f}K"
    else:
        return f"${value:.2f}"

def sales_by_category(df):
    """Calculate sales by category, sorted descending"""
    if 'category' not in df.columns or 'sales' not in df.columns:
        return pd.DataFrame()
    
    return df.groupby('category')['sales'].sum().sort_values(ascending=False).reset_index()

def monthly_sales(df):
    """Calculate monthly sales trend"""
    if 'order_date' not in df.columns or 'sales' not in df.columns:
        return pd.DataFrame()
    
    df_clean = df.dropna(subset=['order_date'])
    monthly = df_clean.groupby(df_clean['order_date'].dt.to_period('M'))['sales'].sum().reset_index()
    monthly['order_date'] = monthly['order_date'].astype(str)
    return monthly

def profit_by_subcategory(df):
    """Calculate profit by sub-category and identify top loss-makers"""
    if 'sub_category' not in df.columns or 'profit' not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    
    profit_data = df.groupby('sub_category')['profit'].sum().sort_values().reset_index()
    loss_makers = profit_data[profit_data['profit'] < 0].head(5)
    
    return profit_data, loss_makers

def top_states(df, n=10):
    """Get top N states by GMV"""
    if 'state' not in df.columns or 'sales' not in df.columns:
        return pd.DataFrame()
    
    return df.groupby('state')['sales'].sum().sort_values(ascending=False).head(n).reset_index()

def top_cities(df, n=10):
    """Get top N cities by GMV"""
    if 'city' not in df.columns or 'sales' not in df.columns:
        return pd.DataFrame()
    
    return df.groupby('city')['sales'].sum().sort_values(ascending=False).head(n).reset_index()

def ship_mode_profit(df):
    """Calculate total profit by shipping mode"""
    if 'ship_mode' not in df.columns or 'profit' not in df.columns:
        return pd.DataFrame()
    
    return df.groupby('ship_mode')['profit'].sum().sort_values(ascending=False).reset_index()

def make_discount_bands(df):
    """Create discount bands for analysis"""
    if 'discount' not in df.columns:
        return df
    
    df = df.copy()
    df['discount_band'] = pd.cut(
        df['discount'], 
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
        labels=['0-10%', '11-20%', '21-30%', '31-40%', '41-50%', '50%+'],
        include_lowest=True
    )
    return df

def band_profit_table(df_band):
    """Calculate average profit per order by discount band"""
    if 'discount_band' not in df_band.columns or 'profit' not in df_band.columns:
        return pd.DataFrame()
    
    band_stats = df_band.groupby('discount_band').agg({
        'profit': 'mean',
        'order_id': 'nunique' if 'order_id' in df_band.columns else 'count'
    }).round(2)
    
    band_stats.columns = ['avg_profit_per_order', 'order_count']
    return band_stats.reset_index()

def simulate_profit(df, band_table, target_discount):
    """Simulate profit based on target discount using band-based repricing"""
    if df.empty or band_table.empty:
        return {'scenario_profit': 0, 'delta_profit': 0, 'scenario_margin': 0}
    
    # Map target discount to nearest band
    band_mapping = {
        '0-10%': 0.05,
        '11-20%': 0.15,
        '21-30%': 0.25,
        '31-40%': 0.35,
        '41-50%': 0.45,
        '50%+': 0.75
    }
    
    # Find closest band to target
    target_decimal = target_discount / 100
    closest_band = min(band_mapping.keys(), 
                      key=lambda x: abs(band_mapping[x] - target_decimal))
    
    # Get average profit per order for the target band
    target_row = band_table[band_table['discount_band'] == closest_band]
    if target_row.empty:
        # Fallback to first available band
        target_avg_profit = band_table['avg_profit_per_order'].iloc[0]
    else:
        target_avg_profit = target_row['avg_profit_per_order'].iloc[0]
    
    # Calculate scenario metrics
    total_orders = df['order_id'].nunique() if 'order_id' in df.columns else len(df)
    scenario_profit = target_avg_profit * total_orders
    actual_profit = df['profit'].sum()
    delta_profit = scenario_profit - actual_profit
    
    actual_gmv = df['sales'].sum()
    scenario_margin = (scenario_profit / actual_gmv * 100) if actual_gmv > 0 else 0
    
    return {
        'scenario_profit': scenario_profit,
        'delta_profit': delta_profit,
        'scenario_margin': scenario_margin,
        'actual_profit': actual_profit
    }

def main():
    # Load and process data
    df_raw = load_data()
    
    if df_raw.empty:
        st.error("Unable to load data. Please ensure superstore.csv exists in the project root.")
        return
    
    df = normalize_columns(df_raw)
    
    if df.empty:
        st.error("No valid data found after processing. Please check the data format.")
        return
    
    # Check for missing critical columns
    required_cols = ['sales', 'profit']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing critical columns: {', '.join(missing_cols)}. Some features may not work properly.")
    
    st.title("Superstore Insights")
    st.markdown("*Executive Dashboard for Retail Performance Analysis*")
    
    # Executive KPIs
    kpis = compute_kpis(df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("GMV", format_currency(kpis['gmv']))
    with col2:
        st.metric("Gross Profit", format_currency(kpis['gross_profit']))
    with col3:
        st.metric("Margin %", f"{kpis['margin_pct']:.1f}%")
    with col4:
        st.metric("Orders", f"{kpis['orders']:,}")
    
    st.divider()
    
    # Sales by Category
    st.subheader("Sales by Category")
    st.markdown("*Revenue performance across product categories*")
    
    category_data = sales_by_category(df)
    if not category_data.empty:
        fig1 = px.bar(
            category_data,
            x='sales',
            y='category',
            orientation='h',
            title="Sales by Category (Descending GMV)"
        )
        fig1.update_layout(height=400, showlegend=False)
        fig1.update_xaxes(tickformat='$,.0f')
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("Category data not available")
    
    st.divider()
    
    # Monthly Sales Trend
    st.subheader("Monthly Sales Trend")
    st.markdown("*Sales performance over time*")
    
    monthly_data = monthly_sales(df)
    if not monthly_data.empty:
        fig2 = px.line(
            monthly_data,
            x='order_date',
            y='sales',
            title="Monthly Sales Trend"
        )
        fig2.update_layout(height=400, showlegend=False)
        fig2.update_yaxes(tickformat='$,.0f')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Time series data not available")
    
    st.divider()
    
    # Profitability by Sub-Category
    st.subheader("Profitability by Sub-Category")
    st.markdown("*Identify profitable segments and loss-making areas*")
    
    profit_data, loss_makers = profit_by_subcategory(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not profit_data.empty:
            fig3 = px.bar(
                profit_data.tail(20),  # Show bottom 20 to highlight loss-makers
                x='profit',
                y='sub_category',
                orientation='h',
                title="Total Profit by Sub-Category (Ascending)"
            )
            fig3.update_layout(height=500, showlegend=False)
            fig3.update_xaxes(tickformat='$,.0f')
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Sub-category data not available")
    
    with col2:
        st.markdown("**Top 5 Loss-Making Sub-Categories**")
        if not loss_makers.empty:
            for idx, row in loss_makers.iterrows():
                st.markdown(f"• {row['sub_category']}: {format_currency(row['profit'])}")
        else:
            st.info("No loss-making sub-categories found")
    
    st.divider()
    
    # Geography Leaders
    st.subheader("Geography Leaders")
    st.markdown("*Top performing states and cities by revenue*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        states_data = top_states(df)
        if not states_data.empty:
            fig4 = px.bar(
                states_data,
                x='sales',
                y='state',
                orientation='h',
                title="Top 10 States by GMV"
            )
            fig4.update_layout(height=400, showlegend=False)
            fig4.update_xaxes(tickformat='$,.0f')
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("State data not available")
    
    with col2:
        cities_data = top_cities(df)
        if not cities_data.empty:
            fig5 = px.bar(
                cities_data,
                x='sales',
                y='city',
                orientation='h',
                title="Top 10 Cities by GMV"
            )
            fig5.update_layout(height=400, showlegend=False)
            fig5.update_xaxes(tickformat='$,.0f')
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.warning("City data not available")
    
    st.divider()
    
    # Shipping Mode Profit
    st.subheader("Shipping Mode Profit")
    st.markdown("*Profitability analysis by shipping method*")
    
    shipping_data = ship_mode_profit(df)
    if not shipping_data.empty:
        fig6 = px.bar(
            shipping_data,
            x='ship_mode',
            y='profit',
            title="Total Profit by Shipping Mode"
        )
        fig6.update_layout(height=400, showlegend=False)
        fig6.update_yaxes(tickformat='$,.0f')
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.warning("Shipping mode data not available")
    
    st.divider()
    
    # What-If Discount Simulator
    st.subheader("What-If Discount Simulator")
    st.markdown("*Simulate profitability under different discount policies*")
    
    # Prepare discount band analysis
    df_bands = make_discount_bands(df)
    band_table = band_profit_table(df_bands)
    
    if not band_table.empty:
        # Center the slider
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            target_discount = st.slider(
                "Target Discount (%)",
                min_value=0,
                max_value=50,
                value=15,
                step=1
            )
        
        # Run simulation
        simulation = simulate_profit(df, band_table, target_discount)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Scenario Profit", format_currency(simulation['scenario_profit']))
        
        with col2:
            delta_formatted = format_currency(abs(simulation['delta_profit']))
            delta_symbol = "+" if simulation['delta_profit'] >= 0 else "-"
            st.metric("Δ vs Actual", f"{delta_symbol}{delta_formatted}")
        
        with col3:
            st.metric("Scenario Margin %", f"{simulation['scenario_margin']:.1f}%")
        
        # Comparison chart
        comparison_data = pd.DataFrame({
            'Scenario': ['Actual', 'Target'],
            'Profit': [simulation['actual_profit'], simulation['scenario_profit']]
        })
        
        fig7 = px.bar(
            comparison_data,
            x='Scenario',
            y='Profit',
            title="Actual vs Scenario Profit Comparison"
        )
        fig7.update_layout(height=300, showlegend=False)
        fig7.update_yaxes(tickformat='$,.0f')
        st.plotly_chart(fig7, use_container_width=True)
        
        # Plain English takeaway
        impact = "increase" if simulation['delta_profit'] >= 0 else "decrease"
        takeaway = f"At {target_discount}% target discount, projected profit is {format_currency(simulation['scenario_profit'])} ({impact} vs actual)."
        
        if not loss_makers.empty:
            worst_category = loss_makers.iloc[0]['sub_category']
            takeaway += f" Loss pressure comes mainly from {worst_category}; consider caps."
        
        st.info(takeaway)
        
    else:
        st.warning("Unable to create discount simulation. Discount data may be missing.")

if __name__ == "__main__":
    main()