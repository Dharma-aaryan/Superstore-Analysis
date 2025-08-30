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

# Dark mode professional color palette
COLORS = {
    'sales': '#00d4ff',        # Bright blue for sales
    'profit_positive': '#00ff88',  # Bright green for positive profit
    'profit_negative': '#ff4b4b',  # Bright red for losses
    'neutral': '#a0a0a0',      # Light gray for neutral/secondary
    'accent': '#ff6b6b',       # Coral for highlights
    'background': '#0e1117',   # Dark background
    'card_bg': '#262730'       # Card background
}

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

def create_kpi_card(title, value, description, color):
    """Create a styled KPI card with dark mode colors"""
    card_html = f"""
    <div style="
        background: {color};
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        color: white;
        margin-bottom: 1rem;
        border-left: 4px solid rgba(255,255,255,0.3);
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    ">
        <div style="margin-bottom: 0.5rem;">
            <h4 style="margin: 0; font-size: 0.9rem; color: rgba(255,255,255,0.9);">{title}</h4>
        </div>
        <div style="font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;">{value}</div>
        <p style="margin: 0; font-size: 0.8rem; color: rgba(255,255,255,0.8); line-height: 1.3;">{description}</p>
    </div>
    """
    return card_html

def executive_summary_tab(df):
    """Executive Summary Dashboard"""
    st.header("Executive Summary")
    
    # Metrics explanation
    st.markdown("""
    **Executive Metrics Overview:** These key performance indicators provide a high-level view of business health. 
    GMV shows total revenue scale, Gross Profit indicates profitability, Profit Margin measures efficiency, 
    and Total Orders reflects business volume and customer engagement.
    """)
    
    st.divider()
    
    # KPIs with styled cards
    kpis = compute_kpis(df)
    
    st.subheader("Key Performance Indicators")
    st.markdown("*Core business metrics that drive strategic decisions*")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            create_kpi_card(
                "GMV (Gross Merchandise Value)",
                format_currency(kpis['gmv']),
                "Total revenue generated from all sales transactions. Measures market size and business scale.",
                COLORS['sales']
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            create_kpi_card(
                "Gross Profit",
                format_currency(kpis['gross_profit']),
                "Total profit after direct costs. Key indicator of business profitability and operational efficiency.",
                COLORS['profit_positive']
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            create_kpi_card(
                "Profit Margin",
                f"{kpis['margin_pct']:.1f}%",
                "Profit as percentage of sales. Measures pricing effectiveness and cost management efficiency.",
                COLORS['accent']
            ),
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            create_kpi_card(
                "Total Orders",
                f"{kpis['orders']:,}",
                "Number of unique transactions. Indicates customer engagement and business volume.",
                COLORS['neutral']
            ),
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # Enhanced overview charts with better alignment
    st.subheader("Business Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Container for chart and table alignment
        chart_container = st.container()
        table_container = st.container()
        
        with chart_container:
            st.markdown("#### Sales Performance by Category")
            category_data = sales_by_category(df)
            if not category_data.empty:
                # Enhanced donut chart with consistent colors
                fig1 = px.pie(
                    category_data,
                    values='sales',
                    names='category',
                    title="Revenue Distribution Across Categories",
                    hole=0.4,
                    color_discrete_sequence=[COLORS['sales'], COLORS['accent'], COLORS['neutral'], '#9467bd', '#8c564b']
                )
                fig1.update_traces(textposition='inside', textinfo='percent+label')
                fig1.update_layout(height=400, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2))
                st.plotly_chart(fig1, use_container_width=True)
        
        with table_container:
            if not category_data.empty:
                # Aligned summary table
                category_data['Percentage'] = (category_data['sales'] / category_data['sales'].sum() * 100).round(1)
                category_data['Sales_Formatted'] = category_data['sales'].apply(format_currency)
                
                st.markdown("**Category Performance Summary**")
                st.dataframe(
                    category_data[['category', 'Sales_Formatted', 'Percentage']].rename(columns={
                        'category': 'Category',
                        'Sales_Formatted': 'Sales',
                        'Percentage': 'Share (%)'
                    }),
                    hide_index=True,
                    use_container_width=True,
                    height=150
                )
            else:
                st.warning("Category data not available")
    
    with col2:
        # Container for chart and metrics alignment
        chart_container = st.container()
        metrics_container = st.container()
        
        with chart_container:
            st.markdown("#### Monthly Sales Trend Analysis")
            monthly_data = monthly_sales(df)
            if not monthly_data.empty:
                # Enhanced line chart with area fill
                fig2 = px.area(
                    monthly_data,
                    x='order_date',
                    y='sales',
                    title="Sales Growth Trajectory Over Time",
                    color_discrete_sequence=[COLORS['sales']]
                )
                fig2.update_layout(height=400, showlegend=False)
                fig2.update_yaxes(tickformat='$,.0f')
                st.plotly_chart(fig2, use_container_width=True)
        
        with metrics_container:
            if not monthly_data.empty and len(monthly_data) >= 2:
                # Aligned growth metrics
                latest_month = monthly_data.iloc[-1]['sales']
                previous_month = monthly_data.iloc[-2]['sales']
                growth_rate = ((latest_month - previous_month) / previous_month * 100) if previous_month != 0 else 0
                
                st.markdown("**Monthly Performance Metrics**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(
                        f"""<div style="background: {COLORS['card_bg']}; padding: 1rem; border-radius: 5px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
                        <strong>Latest Month</strong><br>
                        <span style="font-size: 1.2rem; color: {COLORS['sales']};">{format_currency(latest_month)}</span>
                        </div>""",
                        unsafe_allow_html=True
                    )
                with col_b:
                    growth_color = COLORS['profit_positive'] if growth_rate >= 0 else COLORS['profit_negative']
                    st.markdown(
                        f"""<div style="background: {COLORS['card_bg']}; padding: 1rem; border-radius: 5px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
                        <strong>MoM Growth</strong><br>
                        <span style="font-size: 1.2rem; color: {growth_color};">{growth_rate:.1f}%</span>
                        </div>""",
                        unsafe_allow_html=True
                    )
            else:
                st.warning("Time series data not available")

def sales_analysis_tab(df):
    """Sales Analysis Dashboard"""
    st.header("Sales Analysis")
    st.markdown("""**Sales Performance Metrics:** This section analyzes revenue patterns across product categories and time periods. 
    The charts show category-wise revenue distribution and monthly sales trends to identify growth opportunities and seasonal patterns.""")
    
    # Main charts section
    col1, col2 = st.columns(2)
    
    with col1:
        chart_container = st.container()
        with chart_container:
            st.subheader("Sales by Category")
            category_data = sales_by_category(df)
            if not category_data.empty:
                fig = px.bar(
                    category_data,
                    x='sales',
                    y='category',
                    orientation='h',
                    title="Revenue by Product Category",
                    color_discrete_sequence=[COLORS['sales']]
                )
                fig.update_layout(height=400, showlegend=False)
                fig.update_xaxes(tickformat='$,.0f')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Category data not available")
    
    with col2:
        chart_container = st.container()
        with chart_container:
            st.subheader("Monthly Sales Trend")
            monthly_data = monthly_sales(df)
            if not monthly_data.empty:
                fig = px.line(
                    monthly_data,
                    x='order_date',
                    y='sales',
                    title="Sales Performance Over Time",
                    markers=True,
                    color_discrete_sequence=[COLORS['sales']]
                )
                fig.update_layout(height=400, showlegend=False)
                fig.update_yaxes(tickformat='$,.0f')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Time series data not available")
    
    st.divider()
    
    # Aligned sales performance table
    if 'category' in df.columns and 'sub_category' in df.columns:
        st.subheader("Top Performing Sub-Categories")
        st.markdown("*Best performing products by total sales revenue*")
        
        # Get top 10 sub-categories by sales
        sales_summary = df.groupby(['category', 'sub_category']).agg({
            'sales': 'sum',
            'profit': 'sum',
            'order_id': 'nunique' if 'order_id' in df.columns else 'count'
        }).round(2)
        sales_summary.columns = ['Total Sales', 'Total Profit', 'Number of Orders']
        sales_summary = sales_summary.reset_index()
        sales_summary = sales_summary.sort_values('Total Sales', ascending=False).head(10)
        
        # Format the table with better styling
        sales_summary['Total Sales'] = sales_summary['Total Sales'].apply(format_currency)
        sales_summary['Total Profit'] = sales_summary['Total Profit'].apply(format_currency)
        
        # Display top performers in a highlighted section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(
                sales_summary,
                use_container_width=True,
                hide_index=True,
                height=400
            )
        
        with col2:
            st.markdown("**Top Performers**")
            top_5 = sales_summary.head(5)
            for idx, row in top_5.iterrows():
                st.markdown(
                    f"""<div style="background: {COLORS['card_bg']}; border: 1px solid rgba(255,255,255,0.1); padding: 0.5rem; margin: 0.2rem 0; border-radius: 5px; border-left: 3px solid {COLORS['sales']};">
                    <strong>{row['sub_category']}</strong><br>
                    <small>{row['category']}</small><br>
                    <span style="color: {COLORS['sales']};">{row['Total Sales']}</span>
                    </div>""",
                    unsafe_allow_html=True
                )

def profitability_tab(df):
    """Profitability Analysis Dashboard"""
    st.header("Profitability Analysis")
    st.markdown("""**Profitability Insights:** These metrics identify which products and categories generate the highest profits versus losses. 
    Focus areas include sub-category profitability analysis and margin performance to optimize product mix and pricing strategies.""")
    
    profit_data, loss_makers = profit_by_subcategory(df)
    
    # Main analysis section with better alignment
    col1, col2 = st.columns([2, 1])
    
    with col1:
        chart_container = st.container()
        with chart_container:
            st.subheader("Profit by Sub-Category")
            if not profit_data.empty:
                # Enhanced profit visualization with color coding
                fig = px.bar(
                    profit_data.tail(20),  # Show bottom 20 to highlight issues
                    x='profit',
                    y='sub_category',
                    orientation='h',
                    title="Profitability Analysis by Sub-Category",
                    color=profit_data.tail(20)['profit'],
                    color_continuous_scale=[COLORS['profit_negative'], '#ffc107', COLORS['profit_positive']]
                )
                fig.update_layout(height=600, showlegend=False)
                fig.update_xaxes(tickformat='$,.0f')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Sub-category data not available")
    
    with col2:
        # Aligned sidebar content
        sidebar_container = st.container()
        with sidebar_container:
            st.subheader("Loss Analysis")
            if not loss_makers.empty:
                st.markdown("**Top 5 Loss-Making Categories**")
                
                # Styled loss makers display
                for idx, row in loss_makers.iterrows():
                    loss_amount = format_currency(abs(row['profit']))
                    st.markdown(
                        f"""<div style="background: {COLORS['card_bg']}; border: 1px solid rgba(255,255,255,0.1); padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid {COLORS['profit_negative']};">
                        <strong style="color: {COLORS['profit_negative']};">{row['sub_category']}</strong><br>
                        <span style="color: {COLORS['profit_negative']}; font-size: 1.1rem; font-weight: bold;">-{loss_amount}</span>
                        </div>""",
                        unsafe_allow_html=True
                    )
                
                # Total loss metric card
                total_loss = abs(loss_makers['profit'].sum())
                st.markdown(
                    f"""<div style="background: {COLORS['profit_negative']}; color: white; padding: 1.5rem; border-radius: 10px; text-align: center; margin-top: 1rem;">
                    <h4 style="margin: 0; color: white;">Total Loss Impact</h4>
                    <div style="font-size: 1.8rem; font-weight: bold; margin-top: 0.5rem;">{format_currency(total_loss)}</div>
                    <small>From top 5 loss-makers</small>
                    </div>""",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """<div style="background: #d4edda; color: #155724; padding: 1.5rem; border-radius: 10px; text-align: center;">
                    <h4 style="margin: 0;">✅ Excellent Performance</h4>
                    <p style="margin: 0.5rem 0 0 0;">No loss-making sub-categories found</p>
                    </div>""",
                    unsafe_allow_html=True
                )
    
    st.divider()
    
    # Aligned profit margin analysis
    if 'category' in df.columns:
        st.subheader("Profit Margin Analysis by Category")
        st.markdown("*Category-wise profitability comparison*")
        
        margin_data = df.groupby('category').agg({
            'sales': 'sum',
            'profit': 'sum'
        })
        margin_data['profit_margin'] = (margin_data['profit'] / margin_data['sales'] * 100).round(2)
        margin_data = margin_data.reset_index().sort_values('profit_margin', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                margin_data,
                x='category',
                y='profit_margin',
                title="Profit Margin by Category",
                color='profit_margin',
                color_continuous_scale=[COLORS['profit_negative'], '#ffc107', COLORS['profit_positive']]
            )
            fig.update_layout(height=400, showlegend=False)
            fig.update_yaxes(title="Profit Margin (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Margin Performance**")
            # Format and display margin data
            margin_display = margin_data.copy()
            margin_display['sales'] = margin_display['sales'].apply(format_currency)
            margin_display['profit'] = margin_display['profit'].apply(format_currency)
            margin_display = margin_display.rename(columns={
                'category': 'Category',
                'sales': 'Sales',
                'profit': 'Profit',
                'profit_margin': 'Margin (%)'
            })
            
            # Show only categories with data (remove any blank rows)
            margin_display_clean = margin_display.dropna()
            st.dataframe(
                margin_display_clean,
                hide_index=True,
                use_container_width=True
            )

def geography_tab(df):
    """Geography Analysis Dashboard"""
    st.header("Geographic Performance")
    st.markdown("""**Geographic Revenue Analysis:** These metrics show revenue performance across states and cities to identify 
    high-performing markets and expansion opportunities. Rankings help prioritize resource allocation and market focus.""")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # States analysis with aligned chart and table
        chart_container = st.container()
        table_container = st.container()
        
        with chart_container:
            st.subheader("Top States by Revenue")
            states_data = top_states(df)
            if not states_data.empty:
                fig = px.bar(
                    states_data,
                    x='state',
                    y='sales',
                    title="Top 10 States by GMV",
                    color_discrete_sequence=[COLORS['accent']]
                )
                fig.update_layout(height=450, showlegend=False)
                fig.update_yaxes(tickformat='$,.0f')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        with table_container:
            if not states_data.empty:
                st.markdown("**State Performance Summary**")
                # Styled state data display
                states_data['Sales_Formatted'] = states_data['sales'].apply(format_currency)
                
                # Show top 5 in styled cards
                for idx, row in states_data.head(5).iterrows():
                    rank = idx + 1
                    st.markdown(
                        f"""<div style="background: {COLORS['card_bg']}; border: 1px solid rgba(255,255,255,0.1); padding: 0.8rem; margin: 0.3rem 0; border-radius: 8px; border-left: 4px solid {COLORS['accent']}; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="background: {COLORS['accent']}; color: white; padding: 0.2rem 0.5rem; border-radius: 50%; font-size: 0.8rem; margin-right: 0.5rem;">{rank}</span>
                            <strong>{row['state']}</strong>
                        </div>
                        <span style="color: {COLORS['accent']}; font-weight: bold;">{row['Sales_Formatted']}</span>
                        </div>""",
                        unsafe_allow_html=True
                    )
            else:
                st.warning("State data not available")
    
    with col2:
        # Cities analysis with aligned chart and table
        chart_container = st.container()
        table_container = st.container()
        
        with chart_container:
            st.subheader("Top Cities by Revenue")
            cities_data = top_cities(df)
            if not cities_data.empty:
                fig = px.bar(
                    cities_data,
                    x='city',
                    y='sales',
                    title="Top 10 Cities by GMV",
                    color_discrete_sequence=[COLORS['profit_positive']]
                )
                fig.update_layout(height=450, showlegend=False)
                fig.update_yaxes(tickformat='$,.0f')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        with table_container:
            if not cities_data.empty:
                st.markdown("**City Performance Summary**")
                # Styled city data display
                cities_data['Sales_Formatted'] = cities_data['sales'].apply(format_currency)
                
                # Show top 5 in styled cards
                for idx, row in cities_data.head(5).iterrows():
                    rank = idx + 1
                    st.markdown(
                        f"""<div style="background: {COLORS['card_bg']}; border: 1px solid rgba(255,255,255,0.1); padding: 0.8rem; margin: 0.3rem 0; border-radius: 8px; border-left: 4px solid {COLORS['profit_positive']}; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="background: {COLORS['profit_positive']}; color: white; padding: 0.2rem 0.5rem; border-radius: 50%; font-size: 0.8rem; margin-right: 0.5rem;">{rank}</span>
                            <strong>{row['city']}</strong>
                        </div>
                        <span style="color: {COLORS['profit_positive']}; font-weight: bold;">{row['Sales_Formatted']}</span>
                        </div>""",
                        unsafe_allow_html=True
                    )
            else:
                st.warning("City data not available")

def operations_tab(df):
    """Operations Analysis Dashboard"""
    st.header("Operations Analysis")
    st.markdown("""**Operational Efficiency Metrics:** These indicators track shipping mode profitability, order volumes, and customer behavior patterns. 
    Key metrics include average order size, total items sold, and orders per customer to optimize operational processes.""")
    
    # Shipping mode analysis with better alignment
    st.subheader("Shipping Mode Profitability")
    shipping_data = ship_mode_profit(df)
    if not shipping_data.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            chart_container = st.container()
            with chart_container:
                fig = px.bar(
                    shipping_data,
                    x='ship_mode',
                    y='profit',
                    title="Total Profit by Shipping Mode",
                    color='profit',
                    color_continuous_scale=[COLORS['profit_negative'], '#ffc107', COLORS['profit_positive']]
                )
                fig.update_layout(height=400, showlegend=False)
                fig.update_yaxes(tickformat='$,.0f')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Aligned shipping summary with styled cards
            table_container = st.container()
            with table_container:
                st.markdown("**Shipping Performance**")
                
                for idx, row in shipping_data.iterrows():
                    profit_color = COLORS['profit_positive'] if row['profit'] >= 0 else COLORS['profit_negative']
                    profit_formatted = format_currency(abs(row['profit']))
                    profit_sign = "" if row['profit'] >= 0 else "-"
                    
                    st.markdown(
                        f"""<div style="background: {COLORS['card_bg']}; border: 1px solid rgba(255,255,255,0.1); padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid {profit_color};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong>{row['ship_mode']}</strong>
                            <span style="color: {profit_color}; font-weight: bold; font-size: 1.1rem;">{profit_sign}{profit_formatted}</span>
                        </div>
                        </div>""",
                        unsafe_allow_html=True
                    )
    else:
        st.warning("Shipping mode data not available")
    
    st.divider()
    
    # Enhanced operational metrics with styled cards
    if 'quantity' in df.columns and 'order_id' in df.columns:
        st.subheader("Operational Efficiency Metrics")
        st.markdown("*Key operational performance indicators*")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_order_size = df['quantity'].mean()
            st.markdown(
                f"""<div style="background: {COLORS['accent']}; padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <div style="font-size: 1rem; margin-bottom: 0.5rem; color: rgba(255,255,255,0.9);">Average Order Size</div>
                <div style="font-size: 2rem; font-weight: bold;">{avg_order_size:.1f}</div>
                <div style="font-size: 0.8rem; color: rgba(255,255,255,0.8);">items per order</div>
                </div>""",
                unsafe_allow_html=True
            )
        
        with col2:
            total_items = df['quantity'].sum()
            st.markdown(
                f"""<div style="background: {COLORS['neutral']}; padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <div style="font-size: 1rem; margin-bottom: 0.5rem; color: rgba(255,255,255,0.9);">Total Items Sold</div>
                <div style="font-size: 2rem; font-weight: bold;">{total_items:,}</div>
                <div style="font-size: 0.8rem; color: rgba(255,255,255,0.8);">total units</div>
                </div>""",
                unsafe_allow_html=True
            )
        
        with col3:
            unique_customers = df['customer_id'].nunique() if 'customer_id' in df.columns else 0
            avg_orders_per_customer = len(df) / unique_customers if unique_customers > 0 else 0
            st.markdown(
                f"""<div style="background: {COLORS['sales']}; padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <div style="font-size: 1rem; margin-bottom: 0.5rem; color: rgba(255,255,255,0.9);">Avg Orders per Customer</div>
                <div style="font-size: 2rem; font-weight: bold;">{avg_orders_per_customer:.1f}</div>
                <div style="font-size: 0.8rem; color: rgba(255,255,255,0.8);">orders per customer</div>
                </div>""",
                unsafe_allow_html=True
            )

def what_if_calculator_tab(df):
    """What-If Discount Calculator"""
    st.header("What-If Discount Calculator")
    st.markdown("""**Discount Impact Simulator:** This tool models how different discount strategies affect overall profitability. 
    Use the slider to test various discount percentages and see projected profit changes, helping optimize pricing strategies.""")
    
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
            st.success(f"**Recommended**: {target_discount}% discount policy could improve profitability")
        else:
            st.error(f"**Caution**: {target_discount}% discount policy may significantly impact profits")
            
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
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Executive Summary",
        "Sales Analysis", 
        "Profitability",
        "Geography",
        "Operations",
        "What-If Calculator"
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