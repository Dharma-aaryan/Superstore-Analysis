import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

def executive_summary_tab(df):
    """Executive Summary Dashboard"""
    st.header("Executive Summary")
    
    # Project Description
    st.markdown("""
    ### About This Dashboard
    This **Superstore Insights Dashboard** provides comprehensive analytics for retail performance optimization. 
    Built using advanced data analytics, it transforms raw transactional data into actionable business intelligence 
    for executive decision-making. The dashboard analyzes sales patterns, profitability drivers, geographic performance, 
    and operational efficiency to identify growth opportunities and operational improvements.
    """)
    
    st.divider()
    
    # KPIs with explanations
    kpis = compute_kpis(df)
    
    st.subheader("Key Performance Indicators")
    st.markdown("*Core business metrics that drive strategic decisions*")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("GMV (Gross Merchandise Value)", format_currency(kpis['gmv']))
        st.caption("**Total revenue** generated from all sales transactions. Measures market size and business scale.")
    
    with col2:
        st.metric("Gross Profit", format_currency(kpis['gross_profit']))
        st.caption("**Total profit** after direct costs. Key indicator of business profitability and operational efficiency.")
    
    with col3:
        st.metric("Profit Margin", f"{kpis['margin_pct']:.1f}%")
        st.caption("**Profit as percentage of sales**. Measures pricing effectiveness and cost management efficiency.")
    
    with col4:
        st.metric("Total Orders", f"{kpis['orders']:,}")
        st.caption("**Number of unique transactions**. Indicates customer engagement and business volume.")
    
    st.divider()
    
    # Enhanced overview charts
    st.subheader("Business Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Sales Performance by Category")
        category_data = sales_by_category(df)
        if not category_data.empty:
            # Enhanced donut chart
            fig1 = px.pie(
                category_data,
                values='sales',
                names='category',
                title="Revenue Distribution Across Categories",
                hole=0.4
            )
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Summary table
            category_data['Percentage'] = (category_data['sales'] / category_data['sales'].sum() * 100).round(1)
            category_data['Sales_Formatted'] = category_data['sales'].apply(format_currency)
            st.dataframe(
                category_data[['category', 'Sales_Formatted', 'Percentage']].rename(columns={
                    'category': 'Category',
                    'Sales_Formatted': 'Sales',
                    'Percentage': 'Share (%)'
                }),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("Category data not available")
    
    with col2:
        st.markdown("#### Monthly Sales Trend Analysis")
        monthly_data = monthly_sales(df)
        if not monthly_data.empty:
            # Enhanced line chart with area fill
            fig2 = px.area(
                monthly_data,
                x='order_date',
                y='sales',
                title="Sales Growth Trajectory Over Time"
            )
            fig2.update_layout(height=400)
            fig2.update_yaxes(tickformat='$,.0f')
            st.plotly_chart(fig2, use_container_width=True)
            
            # Growth metrics
            if len(monthly_data) >= 2:
                latest_month = monthly_data.iloc[-1]['sales']
                previous_month = monthly_data.iloc[-2]['sales']
                growth_rate = ((latest_month - previous_month) / previous_month * 100) if previous_month != 0 else 0
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Latest Month Sales", format_currency(latest_month))
                with col_b:
                    st.metric("Month-over-Month Growth", f"{growth_rate:.1f}%")
        else:
            st.warning("Time series data not available")

def sales_analysis_tab(df):
    """Sales Analysis Dashboard"""
    st.header("Sales Analysis")
    st.markdown("*Deep dive into sales patterns and category performance*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales by Category")
        category_data = sales_by_category(df)
        if not category_data.empty:
            fig = px.bar(
                category_data,
                x='sales',
                y='category',
                orientation='h',
                title="Revenue by Product Category",
                color='sales',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400, showlegend=False)
            fig.update_xaxes(tickformat='$,.0f')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Category data not available")
    
    with col2:
        st.subheader("Monthly Sales Trend")
        monthly_data = monthly_sales(df)
        if not monthly_data.empty:
            fig = px.line(
                monthly_data,
                x='order_date',
                y='sales',
                title="Sales Performance Over Time",
                markers=True
            )
            fig.update_layout(height=400)
            fig.update_yaxes(tickformat='$,.0f')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Time series data not available")
    
    # Sales performance table
    if 'category' in df.columns and 'sub_category' in df.columns:
        st.subheader("Detailed Sales Performance")
        sales_summary = df.groupby(['category', 'sub_category']).agg({
            'sales': ['sum', 'mean', 'count']
        }).round(2)
        sales_summary.columns = ['Total Sales', 'Avg Sale Value', 'Number of Orders']
        sales_summary = sales_summary.reset_index()
        sales_summary = sales_summary.sort_values('Total Sales', ascending=False)
        
        # Format the table
        sales_summary['Total Sales'] = sales_summary['Total Sales'].apply(format_currency)
        sales_summary['Avg Sale Value'] = sales_summary['Avg Sale Value'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(sales_summary, use_container_width=True, hide_index=True)

def profitability_tab(df):
    """Profitability Analysis Dashboard"""
    st.header("Profitability Analysis")
    st.markdown("*Identify profitable segments and loss-making areas*")
    
    profit_data, loss_makers = profit_by_subcategory(df)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Profit by Sub-Category")
        if not profit_data.empty:
            # Enhanced profit visualization with color coding
            colors = ['red' if x < 0 else 'green' for x in profit_data['profit']]
            fig = px.bar(
                profit_data.tail(20),  # Show bottom 20 to highlight issues
                x='profit',
                y='sub_category',
                orientation='h',
                title="Profitability Analysis by Sub-Category",
                color=profit_data.tail(20)['profit'],
                color_continuous_scale=['red', 'yellow', 'green']
            )
            fig.update_layout(height=600, showlegend=False)
            fig.update_xaxes(tickformat='$,.0f')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Sub-category data not available")
    
    with col2:
        st.subheader("Loss-Making Categories")
        if not loss_makers.empty:
            st.markdown("**Top 5 Loss-Making Sub-Categories:**")
            
            # Create a detailed loss table
            loss_table = loss_makers.copy()
            loss_table['Loss Amount'] = loss_table['profit'].apply(lambda x: format_currency(abs(x)))
            loss_table = loss_table[['sub_category', 'Loss Amount']].rename(columns={
                'sub_category': 'Sub-Category'
            })
            
            st.dataframe(loss_table, hide_index=True, use_container_width=True)
            
            # Loss summary
            total_loss = abs(loss_makers['profit'].sum())
            st.metric("Total Loss from Top 5", format_currency(total_loss))
        else:
            st.success("No loss-making sub-categories found")
    
    # Profit margin analysis
    if 'category' in df.columns:
        st.subheader("Profit Margin Analysis by Category")
        margin_data = df.groupby('category').agg({
            'sales': 'sum',
            'profit': 'sum'
        })
        margin_data['profit_margin'] = (margin_data['profit'] / margin_data['sales'] * 100).round(2)
        margin_data = margin_data.reset_index().sort_values('profit_margin', ascending=False)
        
        fig = px.bar(
            margin_data,
            x='category',
            y='profit_margin',
            title="Profit Margin by Category",
            color='profit_margin',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        fig.update_yaxes(title="Profit Margin (%)")
        st.plotly_chart(fig, use_container_width=True)

def geography_tab(df):
    """Geography Analysis Dashboard"""
    st.header("Geographic Performance")
    st.markdown("*Regional sales leaders and market penetration analysis*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top States by Revenue")
        states_data = top_states(df)
        if not states_data.empty:
            fig = px.bar(
                states_data,
                x='sales',
                y='state',
                orientation='h',
                title="Top 10 States by GMV",
                color='sales',
                color_continuous_scale='blues'
            )
            fig.update_layout(height=500, showlegend=False)
            fig.update_xaxes(tickformat='$,.0f')
            st.plotly_chart(fig, use_container_width=True)
            
            # Top states table
            states_data['Sales_Formatted'] = states_data['sales'].apply(format_currency)
            st.dataframe(
                states_data[['state', 'Sales_Formatted']].rename(columns={
                    'state': 'State',
                    'Sales_Formatted': 'Revenue'
                }),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("State data not available")
    
    with col2:
        st.subheader("Top Cities by Revenue")
        cities_data = top_cities(df)
        if not cities_data.empty:
            fig = px.bar(
                cities_data,
                x='sales',
                y='city',
                orientation='h',
                title="Top 10 Cities by GMV",
                color='sales',
                color_continuous_scale='greens'
            )
            fig.update_layout(height=500, showlegend=False)
            fig.update_xaxes(tickformat='$,.0f')
            st.plotly_chart(fig, use_container_width=True)
            
            # Top cities table
            cities_data['Sales_Formatted'] = cities_data['sales'].apply(format_currency)
            st.dataframe(
                cities_data[['city', 'Sales_Formatted']].rename(columns={
                    'city': 'City',
                    'Sales_Formatted': 'Revenue'
                }),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("City data not available")

def operations_tab(df):
    """Operations Analysis Dashboard"""
    st.header("Operations Analysis")
    st.markdown("*Shipping performance and operational efficiency metrics*")
    
    # Shipping mode analysis
    st.subheader("Shipping Mode Profitability")
    shipping_data = ship_mode_profit(df)
    if not shipping_data.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                shipping_data,
                x='ship_mode',
                y='profit',
                title="Total Profit by Shipping Mode",
                color='profit',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400, showlegend=False)
            fig.update_yaxes(tickformat='$,.0f')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Shipping summary table
            shipping_data['Profit_Formatted'] = shipping_data['profit'].apply(format_currency)
            st.dataframe(
                shipping_data[['ship_mode', 'Profit_Formatted']].rename(columns={
                    'ship_mode': 'Shipping Mode',
                    'Profit_Formatted': 'Total Profit'
                }),
                hide_index=True,
                use_container_width=True
            )
    else:
        st.warning("Shipping mode data not available")
    
    # Additional operational metrics
    if 'quantity' in df.columns and 'order_id' in df.columns:
        st.subheader("Operational Efficiency Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_order_size = df['quantity'].mean()
            st.metric("Average Order Size", f"{avg_order_size:.1f} items")
        
        with col2:
            total_items = df['quantity'].sum()
            st.metric("Total Items Sold", f"{total_items:,}")
        
        with col3:
            unique_customers = df['customer_id'].nunique() if 'customer_id' in df.columns else 0
            avg_orders_per_customer = len(df) / unique_customers if unique_customers > 0 else 0
            st.metric("Avg Orders per Customer", f"{avg_orders_per_customer:.1f}")

def what_if_calculator_tab(df):
    """What-If Discount Calculator"""
    st.header("What-If Discount Calculator")
    st.markdown("*Simulate profitability under different discount policies*")
    
    # Prepare discount band analysis
    df_bands = make_discount_bands(df)
    band_table = band_profit_table(df_bands)
    
    if not band_table.empty:
        # Enhanced calculator interface
        st.subheader("Discount Policy Simulator")
        st.markdown("Adjust the target discount percentage to see projected impact on profitability.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            target_discount = st.slider(
                "Target Discount (%)",
                min_value=0,
                max_value=50,
                value=15,
                step=1,
                help="Set the discount percentage you want to analyze"
            )
        
        st.divider()
        
        # Run simulation
        simulation = simulate_profit(df, band_table, target_discount)
        
        # Enhanced results display
        st.subheader("Simulation Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Profit", format_currency(simulation['actual_profit']))
        
        with col2:
            st.metric("Scenario Profit", format_currency(simulation['scenario_profit']))
        
        with col3:
            delta_formatted = format_currency(abs(simulation['delta_profit']))
            delta_symbol = "+" if simulation['delta_profit'] >= 0 else "-"
            delta_color = "normal" if simulation['delta_profit'] >= 0 else "inverse"
            st.metric("Profit Impact", f"{delta_symbol}{delta_formatted}", 
                     delta=f"{delta_symbol}{delta_formatted}")
        
        with col4:
            st.metric("Scenario Margin", f"{simulation['scenario_margin']:.1f}%")
        
        # Enhanced comparison visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Comparison chart
            comparison_data = pd.DataFrame({
                'Scenario': ['Current', 'Target Discount'],
                'Profit': [simulation['actual_profit'], simulation['scenario_profit']]
            })
            
            fig = px.bar(
                comparison_data,
                x='Scenario',
                y='Profit',
                title=f"Profit Comparison: Current vs {target_discount}% Discount",
                color='Profit',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400, showlegend=False)
            fig.update_yaxes(tickformat='$,.0f')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Discount band analysis table
            st.subheader("Historical Discount Band Performance")
            if not band_table.empty:
                display_table = band_table.copy()
                display_table['avg_profit_per_order'] = display_table['avg_profit_per_order'].apply(
                    lambda x: f"${x:.2f}"
                )
                display_table = display_table.rename(columns={
                    'discount_band': 'Discount Range',
                    'avg_profit_per_order': 'Avg Profit/Order',
                    'order_count': 'Order Count'
                })
                st.dataframe(display_table, hide_index=True, use_container_width=True)
        
        # Enhanced insights
        st.subheader("Strategic Insights")
        
        # Calculate additional insights
        profit_data, loss_makers = profit_by_subcategory(df)
        
        impact_direction = "increase" if simulation['delta_profit'] >= 0 else "decrease"
        impact_percent = abs(simulation['delta_profit'] / simulation['actual_profit'] * 100) if simulation['actual_profit'] != 0 else 0
        
        insights = [
            f"**Target Impact**: At {target_discount}% discount, profit would {impact_direction} by {impact_percent:.1f}% ({format_currency(abs(simulation['delta_profit']))})",
            f"**Margin Analysis**: Target scenario margin of {simulation['scenario_margin']:.1f}% vs current business performance",
            f"**Risk Assessment**: {'Low risk - profit improvement expected' if simulation['delta_profit'] >= 0 else 'High risk - significant profit loss projected'}"
        ]
        
        if not loss_makers.empty:
            worst_category = loss_makers.iloc[0]['sub_category']
            insights.append(f"**Category Focus**: Loss pressure mainly from {worst_category} - consider targeted caps or pricing adjustments")
        
        for insight in insights:
            st.markdown(f"• {insight}")
        
        # Recommendations
        st.subheader("Recommendations")
        if simulation['delta_profit'] >= 0:
            st.success(f"✅ **Recommended**: {target_discount}% discount policy could improve profitability")
        else:
            st.error(f"⚠️ **Caution**: {target_discount}% discount policy may significantly impact profits")
            
    else:
        st.warning("Unable to create discount simulation. Discount data may be missing.")

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
    
    st.title("Superstore Insights Dashboard")
    st.markdown("*Comprehensive Business Analytics for Strategic Decision Making*")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Executive Summary",
        "💰 Sales Analysis", 
        "📈 Profitability",
        "🌍 Geography",
        "🚚 Operations",
        "🧮 What-If Calculator"
    ])
    
    with tab1:
        executive_summary_tab(df)
    
    with tab2:
        sales_analysis_tab(df)
    
    with tab3:
        profitability_tab(df)
    
    with tab4:
        geography_tab(df)
    
    with tab5:
        operations_tab(df)
    
    with tab6:
        what_if_calculator_tab(df)

if __name__ == "__main__":
    main()