import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data import load_superstore, apply_global_filters, get_filter_values, guard_empty
from utils.profit import analyze_profitability_blackholes, get_pareto_chart
from utils.rfm import calculate_rfm, segment_customers, get_rfm_insights
from utils.elasticity import analyze_discount_elasticity, simulate_profit_impact
from utils.basket import perform_market_basket_analysis, get_segment_profitability_matrix
from utils.model import train_profitability_model, get_model_insights
from utils.viz import create_kpi_cards, create_radar_chart, plot_state_map

# Page configuration
st.set_page_config(
    page_title="Superstore Insights Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Superstore Insights Dashboard")
    st.markdown("*Executive-ready business analytics for data-driven decision making*")
    
    # Load preloaded dataset
    try:
        df = load_superstore(None)  # Always use default dataset
        if df is None or df.empty:
            st.error("No data available. Please check the default dataset.")
            return
            
        st.success(f"Data loaded successfully: {len(df):,} records")
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Global filters in sidebar
    st.sidebar.header("Global Filters")
    filter_values = get_filter_values(df)
    
    # Category filter
    categories = st.sidebar.multiselect(
        "Categories",
        options=filter_values['categories'],
        default=filter_values['categories'],
        help="Select product categories to analyze"
    )
    
    # Region filter
    regions = st.sidebar.multiselect(
        "Regions",
        options=filter_values['regions'],
        default=filter_values['regions'],
        help="Select regions to analyze"
    )
    
    # Apply global filters
    filtered_df = apply_global_filters(df, categories, regions)
    
    st.sidebar.success(f"{len(filtered_df):,} records match filters")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Executive Summary", 
        "Profitability Black Holes", 
        "Customer Value (RFM)",
        "Discount Elasticity & Pricing", 
        "Operations & Cross-Sell"
    ])
    
    with tab1:
        executive_summary_tab(filtered_df)
    
    with tab2:
        profitability_blackholes_tab(filtered_df)
    
    with tab3:
        customer_value_tab(filtered_df)
    
    with tab4:
        discount_elasticity_tab(filtered_df)
    
    with tab5:
        operations_crosssell_tab(filtered_df)

def executive_summary_tab(filtered):
    """Executive Summary tab with KPIs and radar chart"""
    guard_empty(filtered)
    
    st.header("Executive Summary")
    st.markdown("*High-level business performance metrics and key profitability concerns*")
    
    # KPI Cards
    kpi_metrics = create_kpi_cards(filtered)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sales", f"${kpi_metrics['total_sales']:,.0f}")
        st.metric("Total Orders", f"{kpi_metrics['total_orders']:,}")
    
    with col2:
        st.metric("Total Profit", f"${kpi_metrics['total_profit']:,.0f}")
        st.metric("Avg Discount", f"{kpi_metrics['avg_discount']:.1%}")
    
    with col3:
        st.metric("Profit Margin", f"{kpi_metrics['profit_margin']:.1%}")
        st.metric("Total Customers", f"{kpi_metrics['total_customers']:,}")
    
    st.divider()
    
    # Geographic Performance Map - Always shown
    st.subheader("Geographic Performance")
    st.markdown("*Profit by State: Total profit generated from all orders in each state*")
    st.markdown("*Profit Margin by State: Average profit margin (profit/sales ratio) for each state*")
    
    # Map options
    col1, col2 = st.columns(2)
    with col1:
        map_metric = st.selectbox("Map Metric", ["Profit", "Profit Margin"], key="exec_map_metric")
    
    if 'state' in filtered.columns:
        if map_metric == "Profit":
            map_fig = plot_state_map(filtered, value_col='profit', title='Total Profit by State')
        else:
            # Create profit margin column - handle division by zero
            filtered_copy = filtered.copy()
            filtered_copy['profit_margin'] = np.where(
                filtered_copy['sales'] > 0,
                (filtered_copy['profit'] / filtered_copy['sales']) * 100,
                0
            )
            map_fig = plot_state_map(filtered_copy, value_col='profit_margin', title='Profit Margin % by State')
        
        if map_fig:
            st.plotly_chart(map_fig, use_container_width=True)
        else:
            st.info("Unable to create geographic map with current data selection.")
    else:
        st.info("State information not available in current data.")
    
    st.divider()
    
    # Top 5 Profitability Black Holes bar chart
    st.subheader("Top 5 Profitability Black Holes")
    st.markdown("*Products and sub-categories with the biggest cumulative losses*")
    
    blackhole_data = analyze_profitability_blackholes(filtered, top_n=5)
    if not blackhole_data.empty:
        # Create horizontal bar chart
        import plotly.express as px
        bar_fig = px.bar(
            blackhole_data,
            x='Loss',
            y='Item',
            orientation='h',
            title="Top 5 Loss-Making Items",
            labels={'Loss': 'Loss Amount ($)', 'Item': 'Product/Category'},
            text='Loss',
            color='Loss',
            color_continuous_scale='Reds_r'
        )
        bar_fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        bar_fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            height=400,
            showlegend=False
        )
        st.plotly_chart(bar_fig, use_container_width=True)
        
        # Auto-generated insights
        st.markdown("### Key Insights")
        total_loss = blackhole_data['Loss'].sum()
        worst_item = blackhole_data.iloc[0]
        
        insights = [
            f"• **${abs(total_loss):,.0f}** in cumulative losses from top 5 problem areas",
            f"• **{worst_item['Item']}** is the biggest profit drain at **${abs(worst_item['Loss']):,.0f}** loss",
            f"• These 5 items represent critical areas requiring immediate management attention"
        ]
        
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("No significant profit losses identified in current data selection.")

def profitability_blackholes_tab(filtered):
    """Profitability Black Holes analysis tab"""
    guard_empty(filtered)
    
    st.header("Profitability Black Holes")
    st.markdown("*Deep dive into loss drivers and profit optimization opportunities*")
    
    # Simplified filters
    col1, col2 = st.columns(2)
    
    with col1:
        category_filter = st.selectbox(
            "Category Focus",
            options=['All'] + list(filtered['category'].unique()) if 'category' in filtered.columns else ['All'],
            help="Focus analysis on specific category"
        )
    
    with col2:
        region_filter = st.selectbox(
            "Region Focus", 
            options=['All'] + list(filtered['region'].unique()) if 'region' in filtered.columns else ['All'],
            help="Focus analysis on specific region"
        )
    
    # Apply drill filters
    drill_df = filtered.copy()
    
    if category_filter != 'All' and 'category' in drill_df.columns:
        drill_df = drill_df[drill_df['category'] == category_filter]
        
    if region_filter != 'All' and 'region' in drill_df.columns:
        drill_df = drill_df[drill_df['region'] == region_filter]
    
    if drill_df.empty:
        st.warning("No data matches the selected drill filters.")
        return
    
    # Three main metrics analysis
    st.subheader("Performance Analysis")
    
    # Calculate three key metrics
    metrics_data = drill_df.groupby('sub_category').agg({
        'profit': ['sum', 'count'],
        'sales': 'sum',
        'discount': 'mean'
    }).reset_index()
    
    metrics_data.columns = ['Sub_Category', 'Total_Loss', 'Order_Count', 'Total_Sales', 'Avg_Discount']
    
    # Filter for loss-making items only
    loss_items = metrics_data[metrics_data['Total_Loss'] < 0].copy()
    loss_items['Loss_Per_Order'] = abs(loss_items['Total_Loss']) / loss_items['Order_Count']
    loss_items['Loss_Per_Sales_Dollar'] = abs(loss_items['Total_Loss']) / loss_items['Total_Sales']
    
    if not loss_items.empty:
        # Create three separate bar charts for the three metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Total Loss Amount**")
            import plotly.express as px
            fig1 = px.bar(
                loss_items.nlargest(10, 'Total_Loss'),
                x='Total_Loss',
                y='Sub_Category',
                orientation='h',
                title="Total Loss by Sub-Category"
            )
            fig1.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.markdown("**Loss Per Order**")
            fig2 = px.bar(
                loss_items.nlargest(10, 'Loss_Per_Order'),
                x='Loss_Per_Order',
                y='Sub_Category',
                orientation='h',
                title="Loss Per Order"
            )
            fig2.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col3:
            st.markdown("**Loss Per Sales Dollar**")
            fig3 = px.bar(
                loss_items.nlargest(10, 'Loss_Per_Sales_Dollar'),
                x='Loss_Per_Sales_Dollar',
                y='Sub_Category',
                orientation='h',
                title="Loss Per Sales Dollar"
            )
            fig3.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
        
        # Insights
        st.markdown("### Key Insights")
        total_loss = loss_items['Total_Loss'].sum()
        worst_total = loss_items.loc[loss_items['Total_Loss'].idxmin()]
        worst_per_order = loss_items.loc[loss_items['Loss_Per_Order'].idxmax()]
        
        insights = [
            f"• **${abs(total_loss):,.0f}** total losses across {len(loss_items)} sub-categories",
            f"• **{worst_total['Sub_Category']}** has highest total loss: **${abs(worst_total['Total_Loss']):,.0f}**",
            f"• **{worst_per_order['Sub_Category']}** has highest loss per order: **${worst_per_order['Loss_Per_Order']:,.2f}**"
        ]
        
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("No significant loss drivers found in the selected data.")

def customer_value_tab(filtered):
    """Customer Value RFM segmentation tab"""
    guard_empty(filtered)
    
    st.header("Customer Value (RFM Segmentation)")
    st.markdown("*Segment customers by Recency, Frequency, and Monetary value for targeted strategies*")
    
    # Calculate RFM
    rfm_data = calculate_rfm(filtered)
    customer_segments = segment_customers(rfm_data)
    
    if customer_segments.empty:
        st.warning("Unable to calculate RFM scores with current data.")
        return
    
    # RFM segment overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Tier Distribution")
        tier_counts = customer_segments['Tier'].value_counts()
        
        for tier in ['VIP', 'Loyal', 'Promising', 'At-Risk']:
            if tier in tier_counts.index:
                count = tier_counts[tier]
                percentage = (count / len(customer_segments)) * 100
                st.metric(f"{tier} Customers", f"{count:,} ({percentage:.1f}%)")
    
    with col2:
        st.subheader("Revenue by Tier")
        
        # Calculate revenue and profit by tier
        tier_metrics = customer_segments.groupby('Tier').agg({
            'Monetary': 'sum',
            'customer_id': 'count'
        }).round(2)
        tier_metrics.columns = ['Total_Revenue', 'Customer_Count']
        
        for tier in tier_metrics.index:
            revenue = tier_metrics.loc[tier, 'Total_Revenue']
            st.metric(f"{tier} Revenue", f"${revenue:,.0f}")
    
    # RFM Cohort Analysis
    st.subheader("RFM Cohort Analysis")
    
    # Create cohort by Region
    cohort_data = customer_segments.merge(
        filtered[['customer_id', 'region']].drop_duplicates(),
        on='customer_id'
    )
    
    cohort_matrix = cohort_data.groupby(['Tier', 'region']).size().reset_index().rename(columns={0: 'Count'})
    
    if not cohort_matrix.empty:
        import plotly.express as px
        fig = px.bar(
            cohort_matrix,
            x='region',
            y='Count',
            color='Tier',
            title="Customer Tier Distribution by Region",
            labels={'Count': 'Number of Customers', 'region': 'Region'},
            text='Count'
        )
        fig.update_traces(texttemplate='%{text}', textposition='inside')
        fig.update_layout(
            xaxis_title="Region",
            yaxis_title="Number of Customers",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("*Each color represents a different customer tier across regions*")
    
    # RFM insights
    st.markdown("### Customer Value Insights")
    insights = get_rfm_insights(customer_segments)
    for insight in insights:
        st.markdown(f"• {insight}")
    
    # Removed export functionality

def discount_elasticity_tab(filtered):
    """Discount elasticity and pricing analysis tab"""
    guard_empty(filtered)
    
    st.header("Discount Elasticity & Pricing")
    st.markdown("*Analyze the relationship between discounts and profitability to optimize pricing strategies*")
    
    # Elasticity analysis
    elasticity_data = analyze_discount_elasticity(filtered)
    
    if elasticity_data.empty:
        st.warning("Unable to perform elasticity analysis with current data.")
        return
    
    # Elasticity explorer charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Discount vs Profit per Order")
        import plotly.express as px
        
        profit_fig = px.bar(
            elasticity_data,
            x='Discount_Band_Mid',
            y='Median_Profit_Per_Order',
            title="Profit by Discount Band",
            labels={
                'Discount_Band_Mid': 'Discount Rate',
                'Median_Profit_Per_Order': 'Median Profit per Order ($)'
            }
        )
        profit_fig.update_layout(showlegend=False)
        st.plotly_chart(profit_fig, use_container_width=True)
        st.markdown("*Identify the discount sweet spot for maximum profitability*")
    
    with col2:
        st.subheader("Discount vs Sales per Order")
        
        sales_fig = px.bar(
            elasticity_data,
            x='Discount_Band_Mid',
            y='Median_Sales_Per_Order',
            title="Sales Volume by Discount Band",
            labels={
                'Discount_Band_Mid': 'Discount Rate',
                'Median_Sales_Per_Order': 'Median Sales per Order ($)'
            }
        )
        sales_fig.update_layout(showlegend=False)
        st.plotly_chart(sales_fig, use_container_width=True)
        st.markdown("*Track how discounts drive sales volume*")
    
    # What-if simulation
    st.subheader("What-If Discount Simulation")
    st.markdown("*Simulate the impact of capping discounts at different levels*")
    
    max_discount = st.slider(
        "Cap discounts at:",
        min_value=0.0,
        max_value=0.5,
        value=0.3,
        step=0.05,
        format="%.0%",
        help="Simulate profit impact if all discounts were capped at this level"
    )
    
    try:
        simulation_result = simulate_profit_impact(filtered, max_discount)
        
        if simulation_result and 'current_profit' in simulation_result:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Total Profit", f"${simulation_result['current_profit']:,.0f}")
            with col2:
                st.metric("Simulated Profit", f"${simulation_result['simulated_profit']:,.0f}")
            with col3:
                impact = simulation_result['simulated_profit'] - simulation_result['current_profit']
                delta_color = "normal" if impact >= 0 else "inverse"
                st.metric("Profit Impact", f"${impact:,.0f}", delta=f"{impact:,.0f}")
            
            # Additional simulation details
            if 'orders_affected' in simulation_result and simulation_result['orders_affected'] > 0:
                st.info(f"**{simulation_result['orders_affected']}** orders would be affected by this discount cap")
        else:
            st.error("Unable to run simulation with current data")
    except Exception as e:
        st.error(f"Simulation failed: {str(e)}")
        # Fallback simple calculation
        try:
            current_profit = filtered['profit'].sum()
            high_discount_orders = filtered[filtered['discount'] > max_discount]
            if not high_discount_orders.empty:
                potential_savings = high_discount_orders['sales'].sum() * (high_discount_orders['discount'].mean() - max_discount)
                estimated_new_profit = current_profit + potential_savings
            else:
                estimated_new_profit = current_profit
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Total Profit", f"${current_profit:,.0f}")
            with col2:
                st.metric("Estimated New Profit", f"${estimated_new_profit:,.0f}")
            with col3:
                impact = estimated_new_profit - current_profit
                st.metric("Estimated Impact", f"${impact:,.0f}")
        except Exception as fallback_error:
            st.error(f"Unable to perform simulation: {str(fallback_error)}")
    
    # Elasticity insights
    st.markdown("### Pricing Insights")
    
    # Find optimal discount range
    optimal_discount = elasticity_data.loc[elasticity_data['Median_Profit_Per_Order'].idxmax()]
    high_volume_discount = elasticity_data.loc[elasticity_data['Order_Count'].idxmax()]
    
    # Calculate impact for insights
    try:
        simulation_for_insights = simulate_profit_impact(filtered, max_discount)
        if simulation_for_insights and 'current_profit' in simulation_for_insights:
            impact = simulation_for_insights['simulated_profit'] - simulation_for_insights['current_profit']
        else:
            impact = 0
    except Exception:
        impact = 0
    
    insights = [
        f"• **Optimal profitability** achieved at **{optimal_discount['Discount_Band']}** discount range",
        f"• **Highest order volume** occurs at **{high_volume_discount['Discount_Band']}** discount range",
        f"• Capping discounts at **{max_discount:.0%}** would result in **${abs(impact):,.0f}** profit {'gain' if impact > 0 else 'loss'}**"
    ]
    
    for insight in insights:
        st.markdown(insight)
    
    # Removed export functionality

def operations_crosssell_tab(filtered):
    """Operations and cross-sell analysis tab"""
    guard_empty(filtered)
    
    st.header("Operations & Cross-Sell")
    st.markdown("*Operational efficiency analysis and cross-selling opportunities*")
    
    # Shipping efficiency analysis
    st.subheader("Shipping Efficiency Analysis")
    
    shipping_analysis = filtered.groupby(['ship_mode', 'region']).agg({
        'profit': 'mean',
        'order_id': 'count',
        'sales': 'mean'
    }).round(2)
    shipping_analysis.columns = ['Avg_Profit', 'Order_Count', 'Avg_Sales']
    
    # Create shipping profitability scatter plot instead of heatmap
    shipping_data = shipping_analysis.reset_index()
    
    if not shipping_data.empty:
        import plotly.express as px
        fig = px.scatter(
            shipping_data,
            x='Order_Count',
            y='Avg_Profit',
            color='ship_mode',
            size='Avg_Sales',
            hover_data=['region'],
            title="Shipping Profitability Analysis",
            labels={
                'Order_Count': 'Number of Orders',
                'Avg_Profit': 'Average Profit per Order ($)',
                'ship_mode': 'Shipping Mode'
            }
        )
        fig.update_layout(
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("*Bubble size represents average sales. Higher and right is better (more profit, more volume)*")
    
    # Market Basket Analysis
    st.subheader("Market Basket Analysis")
    st.markdown("*Market basket analysis reveals which products are frequently bought together. This helps identify cross-selling opportunities and optimal product placement strategies.*")
    
    with st.expander("Advanced Settings"):
        min_support = st.slider("Minimum Support", 0.01, 0.05, 0.02, 0.01)
        max_len = st.slider("Maximum Rule Length", 2, 5, 2)
    
    try:
        basket_rules = perform_market_basket_analysis(filtered, min_support=min_support, max_len=max_len)
        
        if not basket_rules.empty:
            st.markdown("#### Top Association Rules by Lift")
            
            # Display top rules
            top_rules = basket_rules.nlargest(10, 'lift')[
                ['antecedents', 'consequents', 'support', 'confidence', 'lift']
            ].round(3)
            
            st.dataframe(top_rules, use_container_width=True, hide_index=True)
            st.markdown("*Higher lift values indicate stronger associations between products*")
            
            # Create visualization for market basket rules
            if len(basket_rules) >= 5:
                top_5_rules = basket_rules.nlargest(5, 'lift')
                rule_labels = [f"{list(rule['antecedents'])[0]} → {list(rule['consequents'])[0]}" for _, rule in top_5_rules.iterrows()]
                
                import plotly.express as px
                fig = px.bar(
                    x=top_5_rules['lift'].values,
                    y=rule_labels,
                    orientation='h',
                    title="Top 5 Product Association Rules",
                    labels={'x': 'Lift Score', 'y': 'Product Association Rule'}
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("*Lift > 1 indicates products are bought together more often than by chance*")
        else:
            st.info(f"No association rules found with minimum support of {min_support}. Try lowering the support threshold.")
    
    except Exception as e:
        st.warning(f"Market basket analysis failed: {str(e)}. This may be due to insufficient data or missing dependencies.")
    
    # Segment Profitability Matrix
    st.subheader("Segment Profitability Matrix")
    
    profitability_matrix = get_segment_profitability_matrix(filtered)
    
    if not profitability_matrix.empty:
        # Create bar chart for segment profitability
        if not profitability_matrix.empty:
            import plotly.express as px
            # Create bar chart showing segment profitability
            segment_profit = profitability_matrix.groupby('segment')['Profit_Margin'].mean().reset_index()
            segment_profit = segment_profit.sort_values('Profit_Margin', ascending=True)
            
            fig = px.bar(
                segment_profit,
                x='Profit_Margin',
                y='segment',
                orientation='h',
                title="Average Profit Margin by Customer Segment",
                labels={'Profit_Margin': 'Average Profit Margin (%)', 'segment': 'Customer Segment'}
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("*Shows which customer segments generate the highest profit margins*")
    
    # ML section removed as requested

if __name__ == "__main__":
    main()
