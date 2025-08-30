import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="Superstore Insights Dashboard", 
    page_icon="📊", 
    layout="wide"
)

@st.cache_data
def load_data():
    """Load the preloaded superstore dataset"""
    try:
        if os.path.exists('superstore.csv'):
            df = pd.read_csv('superstore.csv')
        else:
            st.error("Superstore dataset not found. Please ensure superstore.csv is in the project directory.")
            return pd.DataFrame()
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.replace(r'[^a-z0-9]+', '_', regex=True).str.strip('_')
        
        # Handle common column name variations
        column_mapping = {
            'order_date': 'order_date',
            'orderdate': 'order_date', 
            'date': 'order_date',
            'sales': 'sales',
            'revenue': 'sales',
            'profit': 'profit',
            'customer_id': 'customer_id',
            'customerid': 'customer_id',
            'customer_name': 'customer_name',
            'customername': 'customer_name',
            'product_name': 'product_name',
            'productname': 'product_name',
            'category': 'category',
            'sub_category': 'sub_category',
            'subcategory': 'sub_category',
            'region': 'region',
            'state': 'state',
            'city': 'city',
            'discount': 'discount',
            'quantity': 'quantity',
            'ship_mode': 'ship_mode',
            'shipmode': 'ship_mode',
            'segment': 'segment',
            'order_id': 'order_id',
            'orderid': 'order_id'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert data types
        if 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        
        numeric_cols = ['sales', 'profit', 'discount', 'quantity']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna()
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def calculate_rfm(df):
    """Calculate RFM scores for customer segmentation"""
    try:
        if df.empty or 'customer_id' not in df.columns:
            return pd.DataFrame()
        
        reference_date = df['order_date'].max() + timedelta(days=1)
        
        rfm = df.groupby('customer_id').agg({
            'order_date': lambda x: (reference_date - x.max()).days,
            'order_id': 'nunique',
            'sales': 'sum'
        }).round(2)
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # Calculate RFM scores (1-5 scale)
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
        
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        return rfm
    except Exception:
        return pd.DataFrame()

def segment_customers(rfm_df):
    """Segment customers based on RFM scores"""
    if rfm_df.empty:
        return pd.DataFrame()
    
    def assign_segment(row):
        if row['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
            return 'Champions'
        elif row['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
            return 'Loyal'
        elif row['RFM_Score'] in ['512', '511', '422', '421', '412', '411', '311']:
            return 'Potential'
        elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
            return 'New'
        elif row['RFM_Score'] in ['155', '254', '245', '253', '252', '243', '242']:
            return 'Promising'
        elif row['RFM_Score'] in ['331', '321', '231', '241', '251']:
            return 'Need Attention'
        elif row['RFM_Score'] in ['132', '123', '122', '212', '211']:
            return 'About to Sleep'
        elif row['RFM_Score'] in ['121', '131', '141', '151']:
            return 'At Risk'
        elif row['RFM_Score'] in ['155', '154', '245', '244', '253', '252']:
            return 'Cannot Lose Them'
        elif row['RFM_Score'] in ['332', '322', '231', '241', '251', '233', '232']:
            return 'Hibernating'
        else:
            return 'Lost'
    
    rfm_df['Segment'] = rfm_df.apply(assign_segment, axis=1)
    return rfm_df

def main():
    st.title("Superstore Insights Dashboard")
    st.markdown("*Executive-ready business analytics for strategic decision making*")
    
    # Load data
    data = load_data()
    
    if data.empty:
        st.error("No data available. Please check the superstore.csv file.")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    if 'order_date' in data.columns:
        min_date = data['order_date'].min().date()
        max_date = data['order_date'].max().date()
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            data = data[
                (data['order_date'].dt.date >= start_date) & 
                (data['order_date'].dt.date <= end_date)
            ]
    
    # Regional filter
    if 'region' in data.columns:
        unique_regions = data['region'].dropna().unique()
        regions = ['All'] + list(unique_regions)
        selected_region = st.sidebar.selectbox("Region", regions)
        if selected_region != 'All':
            data = data[data['region'] == selected_region]
    
    # Category filter  
    if 'category' in data.columns:
        unique_categories = data['category'].dropna().unique()
        categories = ['All'] + list(unique_categories)
        selected_category = st.sidebar.selectbox("Category", categories)
        if selected_category != 'All':
            data = data[data['category'] == selected_category]
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Executive Summary", 
        "Profitability Analysis", 
        "Customer Segments", 
        "Discount & Pricing", 
        "Operations & Cross-sell"
    ])
    
    with tab1:
        executive_summary_tab(data)
    
    with tab2:
        profitability_analysis_tab(data)
    
    with tab3:
        customer_segments_tab(data)
    
    with tab4:
        discount_pricing_tab(data)
    
    with tab5:
        operations_crosssell_tab(data)

def executive_summary_tab(data):
    """Executive summary with key KPIs"""
    if data.empty:
        st.warning("No data available for the selected filters.")
        return
    
    st.header("Executive Summary")
    st.markdown("*Key performance indicators and business overview*")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = data['sales'].sum()
        st.metric("Total Sales", f"${total_sales:,.0f}")
    
    with col2:
        total_profit = data['profit'].sum()
        profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
        st.metric("Total Profit", f"${total_profit:,.0f}", f"{profit_margin:.1f}% margin")
    
    with col3:
        total_orders = data['order_id'].nunique() if 'order_id' in data.columns else len(data)
        avg_order_value = total_sales / total_orders if total_orders > 0 else 0
        st.metric("Total Orders", f"{total_orders:,}", f"${avg_order_value:.0f} avg")
    
    with col4:
        unique_customers = data['customer_id'].nunique() if 'customer_id' in data.columns else 0
        st.metric("Unique Customers", f"{unique_customers:,}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales by Category")
        if 'category' in data.columns:
            category_sales = data.groupby('category')['sales'].sum().sort_values(ascending=False)
            fig = px.bar(
                x=category_sales.values,
                y=category_sales.index,
                orientation='h',
                title="Sales Performance by Category"
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Monthly Sales Trend")
        if 'order_date' in data.columns:
            monthly_sales = data.groupby(data['order_date'].dt.to_period('M'))['sales'].sum()
            fig = px.line(
                x=monthly_sales.index.astype(str),
                y=monthly_sales.values,
                title="Sales Trend Over Time"
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def profitability_analysis_tab(data):
    """Profitability black holes analysis"""
    if data.empty:
        st.warning("No data available for the selected filters.")
        return
    
    st.header("Profitability Analysis")
    st.markdown("*Identify loss-making segments and improvement opportunities*")
    
    # Calculate loss analysis by sub-category
    if 'sub_category' in data.columns:
        loss_analysis = data.groupby('sub_category').agg({
            'profit': 'sum',
            'sales': 'sum',
            'order_id': 'nunique' if 'order_id' in data.columns else 'count'
        }).round(2)
        
        loss_analysis.columns = ['Total_Profit', 'Total_Sales', 'Order_Count']
        loss_analysis = loss_analysis[loss_analysis['Total_Profit'] < 0].copy()
        
        if not loss_analysis.empty:
            loss_analysis['Loss_Per_Order'] = abs(loss_analysis['Total_Profit']) / loss_analysis['Order_Count']
            loss_analysis['Loss_Per_Sales_Dollar'] = abs(loss_analysis['Total_Profit']) / loss_analysis['Total_Sales']
            
            # Three metrics charts
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Total Loss Amount")
                top_loss = loss_analysis.nsmallest(10, 'Total_Profit')
                fig1 = px.bar(
                    x=abs(top_loss['Total_Profit']),
                    y=top_loss.index,
                    orientation='h',
                    title="Biggest Loss Categories"
                )
                fig1.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.subheader("Loss Per Order")
                top_loss_order = loss_analysis.nlargest(10, 'Loss_Per_Order')
                fig2 = px.bar(
                    x=top_loss_order['Loss_Per_Order'],
                    y=top_loss_order.index,
                    orientation='h',
                    title="Highest Loss Per Order"
                )
                fig2.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
            
            with col3:
                st.subheader("Loss Per Sales Dollar")
                top_loss_sales = loss_analysis.nlargest(10, 'Loss_Per_Sales_Dollar')
                fig3 = px.bar(
                    x=top_loss_sales['Loss_Per_Sales_Dollar'],
                    y=top_loss_sales.index,
                    orientation='h',
                    title="Highest Loss Rate"
                )
                fig3.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)
            
            # Key insights
            st.subheader("Key Insights")
            total_loss = abs(loss_analysis['Total_Profit'].sum())
            worst_category = loss_analysis.loc[loss_analysis['Total_Profit'].idxmin()]
            
            st.markdown(f"• **${total_loss:,.0f}** total losses across {len(loss_analysis)} sub-categories")
            st.markdown(f"• **{worst_category.name}** has the highest total loss: **${abs(worst_category['Total_Profit']):,.0f}**")
            st.markdown(f"• Average loss per order across loss-making categories: **${loss_analysis['Loss_Per_Order'].mean():.2f}**")
        else:
            st.info("No loss-making sub-categories found in the selected data.")
    else:
        st.error("Sub-category data not available for analysis.")

def customer_segments_tab(data):
    """Customer segmentation using RFM analysis"""
    if data.empty:
        st.warning("No data available for the selected filters.")
        return
    
    st.header("Customer Segments (RFM Analysis)")
    st.markdown("*Customer segmentation based on Recency, Frequency, and Monetary value*")
    
    # Calculate RFM
    rfm_data = calculate_rfm(data)
    
    if rfm_data.empty:
        st.error("Unable to calculate RFM scores. Please check data requirements.")
        return
    
    # Segment customers
    segmented_data = segment_customers(rfm_data)
    
    if segmented_data.empty:
        st.error("Unable to segment customers.")
        return
    
    # Segment overview
    segment_summary = segmented_data.groupby('Segment').agg({
        'Monetary': ['count', 'mean', 'sum']
    }).round(2)
    segment_summary.columns = ['Customer_Count', 'Avg_Value', 'Total_Value']
    segment_summary = segment_summary.reset_index()
    
    # Segment distribution chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Distribution by Segment")
        fig1 = px.bar(
            segment_summary,
            x='Customer_Count',
            y='Segment',
            orientation='h',
            title="Number of Customers per Segment"
        )
        fig1.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Revenue by Segment")
        fig2 = px.bar(
            segment_summary,
            x='Total_Value',
            y='Segment',
            orientation='h',
            title="Total Revenue per Segment"
        )
        fig2.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Segment insights
    st.subheader("Segment Insights")
    best_segment = segment_summary.loc[segment_summary['Total_Value'].idxmax()]
    largest_segment = segment_summary.loc[segment_summary['Customer_Count'].idxmax()]
    
    st.markdown(f"• **{best_segment['Segment']}** generates the highest revenue: **${best_segment['Total_Value']:,.0f}**")
    st.markdown(f"• **{largest_segment['Segment']}** has the most customers: **{largest_segment['Customer_Count']:,}**")
    st.markdown(f"• Average customer value across all segments: **${segment_summary['Avg_Value'].mean():.0f}**")

def discount_pricing_tab(data):
    """Discount elasticity and pricing analysis"""
    if data.empty:
        st.warning("No data available for the selected filters.")
        return
    
    st.header("Discount & Pricing Analysis")
    st.markdown("*Analyze discount effectiveness and pricing strategies*")
    
    if 'discount' not in data.columns:
        st.error("Discount data not available for analysis.")
        return
    
    # Create discount bands
    data_copy = data.copy()
    data_copy['Discount_Band'] = pd.cut(
        data_copy['discount'], 
        bins=[0, 0.1, 0.2, 0.3, 0.4, 1.0],
        labels=['0-10pct', '11-20pct', '21-30pct', '31-40pct', '40pct+'],
        include_lowest=True
    )
    
    # Discount analysis
    discount_analysis = data_copy.groupby('Discount_Band').agg({
        'profit': 'mean',
        'sales': 'mean',
        'order_id': 'nunique' if 'order_id' in data_copy.columns else 'count'
    }).round(2)
    discount_analysis.columns = ['Avg_Profit', 'Avg_Sales', 'Order_Count']
    discount_analysis = discount_analysis.reset_index()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Profit by Discount Band")
        fig1 = px.bar(
            discount_analysis,
            x='Discount_Band',
            y='Avg_Profit',
            title="Average Profit by Discount Level"
        )
        fig1.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Order Volume by Discount Band")
        fig2 = px.bar(
            discount_analysis,
            x='Discount_Band',
            y='Order_Count',
            title="Order Volume by Discount Level"
        )
        fig2.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Discount simulation
    st.subheader("Discount Impact Simulation")
    max_discount = st.slider(
        "Cap discounts at:",
        min_value=0.0,
        max_value=0.5,
        value=0.3,
        step=0.05,
        format="%.2f"
    )
    
    # Simple simulation
    current_profit = data['profit'].sum()
    high_discount_orders = data[data['discount'] > max_discount]
    
    if not high_discount_orders.empty:
        # Estimate impact of capping discounts
        potential_savings = high_discount_orders['sales'].sum() * 0.1  # Simple estimate
        simulated_profit = current_profit + potential_savings
        orders_affected = len(high_discount_orders)
    else:
        simulated_profit = current_profit
        orders_affected = 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Profit", f"${current_profit:,.0f}")
    with col2:
        st.metric("Simulated Profit", f"${simulated_profit:,.0f}")
    with col3:
        impact = simulated_profit - current_profit
        st.metric("Estimated Impact", f"${impact:,.0f}")
    
    if orders_affected > 0:
        st.info(f"**{orders_affected}** orders would be affected by this discount cap")

def operations_crosssell_tab(data):
    """Operations and cross-sell analysis"""
    if data.empty:
        st.warning("No data available for the selected filters.")
        return
    
    st.header("Operations & Cross-sell")
    st.markdown("*Operational efficiency and cross-selling opportunities*")
    
    # Shipping analysis
    if 'ship_mode' in data.columns:
        st.subheader("Shipping Mode Analysis")
        shipping_analysis = data.groupby('ship_mode').agg({
            'profit': 'mean',
            'sales': 'mean',
            'order_id': 'nunique' if 'order_id' in data.columns else 'count'
        }).round(2)
        shipping_analysis.columns = ['Avg_Profit', 'Avg_Sales', 'Order_Count']
        shipping_analysis = shipping_analysis.reset_index()
        
        fig = px.bar(
            shipping_analysis,
            x='ship_mode',
            y='Avg_Profit',
            title="Average Profit by Shipping Mode"
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Category performance
    if 'category' in data.columns and 'sub_category' in data.columns:
        st.subheader("Category Performance")
        category_performance = data.groupby(['category', 'sub_category']).agg({
            'sales': 'sum',
            'profit': 'sum'
        }).round(2)
        category_performance = category_performance.reset_index()
        
        # Top performing sub-categories
        top_subcategories = category_performance.nlargest(10, 'sales')
        
        fig = px.bar(
            top_subcategories,
            x='sales',
            y='sub_category',
            color='category',
            orientation='h',
            title="Top 10 Sub-Categories by Sales"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Simple market basket analysis
    st.subheader("Cross-sell Opportunities")
    st.markdown("*Categories frequently purchased together*")
    
    if 'order_id' in data.columns and 'category' in data.columns:
        # Find orders with multiple categories
        order_categories = data.groupby('order_id')['category'].apply(list).reset_index()
        multi_category_orders = order_categories[order_categories['category'].apply(len) > 1]
        
        if not multi_category_orders.empty:
            # Count category combinations
            from itertools import combinations
            category_pairs = []
            for categories in multi_category_orders['category']:
                for pair in combinations(set(categories), 2):
                    category_pairs.append(sorted(pair))
            
            if category_pairs:
                pair_counts = pd.Series(category_pairs).apply(tuple).value_counts().head(10)
                
                pair_df = pd.DataFrame({
                    'Category_Pair': [f"{pair[0]} + {pair[1]}" for pair in pair_counts.index],
                    'Frequency': list(pair_counts.values)
                })
                
                fig = px.bar(
                    pair_df,
                    x='Frequency',
                    y='Category_Pair',
                    orientation='h',
                    title="Most Common Category Combinations"
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"• **{len(multi_category_orders)}** orders contain multiple categories")
                st.markdown(f"• **{pair_counts.iloc[0]}** orders contain the most common category combination")
            else:
                st.info("No significant category combinations found.")
        else:
            st.info("Most orders contain items from a single category.")
    else:
        st.info("Order and category data required for cross-sell analysis.")

if __name__ == "__main__":
    main()