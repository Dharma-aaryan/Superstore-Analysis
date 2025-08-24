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
    st.title("📊 Superstore Insights Dashboard")
    st.markdown("*Executive-ready business analytics for data-driven decision making*")
    
    # File uploader with fallback
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], help="Upload your superstore data or use the default dataset")
    
    # Load and clean data
    try:
        df = load_superstore(uploaded_file)
        if df is None or df.empty:
            st.error("No data available. Please upload a valid CSV file.")
            return
            
        st.success(f"✅ Data loaded successfully: {len(df):,} records")
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Global filters in sidebar
    st.sidebar.header("🎛️ Global Filters")
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
    
    st.sidebar.success(f"📊 {len(filtered_df):,} records match filters")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Executive Summary", 
        "🕳️ Profitability Black Holes", 
        "👥 Customer Value (RFM)",
        "💰 Discount Elasticity & Pricing", 
        "🔄 Operations & Cross-Sell"
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
    
    st.header("📈 Executive Summary")
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
    
    # Choropleth map toggle and display
    show_choropleth = st.toggle("Show choropleth map")
    if show_choropleth and 'state' in filtered.columns:
        st.subheader("🗺️ Geographic Performance")
        
        # Map options
        col1, col2 = st.columns(2)
        with col1:
            map_metric = st.selectbox("Map Metric", ["Profit", "Profit Margin"], key="exec_map_metric")
        
        if map_metric == "Profit":
            map_fig = plot_state_map(filtered, value_col='profit', title='Profit by State')
        else:
            # Create profit margin column
            filtered_copy = filtered.copy()
            filtered_copy['profit_margin'] = filtered_copy['profit'] / filtered_copy['sales'].replace(0, pd.NA)
            map_fig = plot_state_map(filtered_copy, value_col='profit_margin', title='Profit Margin by State')
        
        if map_fig:
            st.plotly_chart(map_fig, use_container_width=True)
    
    st.divider()
    
    # Top 5 Profitability Black Holes radar chart
    st.subheader("🎯 Top 5 Profitability Black Holes")
    st.markdown("*Products and sub-categories with the biggest cumulative losses*")
    
    blackhole_data = analyze_profitability_blackholes(filtered, top_n=5)
    if not blackhole_data.empty:
        radar_fig = create_radar_chart(blackhole_data)
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # Auto-generated insights
        st.markdown("### 🔍 Key Insights")
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
    
    st.header("🕳️ Profitability Black Holes")
    st.markdown("*Deep dive into loss drivers and profit optimization opportunities*")
    
    # Drill filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        category_filter = st.selectbox(
            "Category Focus",
            options=['All'] + list(filtered['category'].unique()) if 'category' in filtered.columns else ['All'],
            help="Focus analysis on specific category"
        )
    
    with col2:
        subcategory_filter = st.selectbox(
            "Sub-Category Focus",
            options=['All'] + list(filtered['sub_category'].unique()) if 'sub_category' in filtered.columns else ['All'],
            help="Focus analysis on specific sub-category"
        )
    
    with col3:
        region_filter = st.selectbox(
            "Region Focus", 
            options=['All'] + list(filtered['region'].unique()) if 'region' in filtered.columns else ['All'],
            help="Focus analysis on specific region"
        )
    
    with col4:
        segment_filter = st.selectbox(
            "Customer Segment",
            options=['All'] + list(filtered['segment'].unique()) if 'segment' in filtered.columns else ['All'],
            help="Focus analysis on specific customer segment"
        )
    
    # Additional filter row
    col5, col6, col7 = st.columns(3)
    
    with col5:
        discount_band = st.selectbox(
            "Discount Band",
            options=['All', '0-10%', '10-20%', '20-30%', '30%+'],
            help="Filter by discount level"
        )
        
    with col6:
        ship_mode_filter = st.selectbox(
            "Ship Mode",
            options=['All'] + list(filtered['ship_mode'].unique()) if 'ship_mode' in filtered.columns else ['All'],
            help="Filter by shipping method"
        )
        
    with col7:
        profit_threshold = st.selectbox(
            "Profit Filter",
            options=['All', 'Loss-making only', 'Profitable only', 'Break-even'],
            help="Filter by profitability status"
        )
    
    # Apply drill filters
    drill_df = filtered.copy()
    
    if category_filter != 'All' and 'category' in drill_df.columns:
        drill_df = drill_df[drill_df['category'] == category_filter]
        
    if subcategory_filter != 'All' and 'sub_category' in drill_df.columns:
        drill_df = drill_df[drill_df['sub_category'] == subcategory_filter]
        
    if region_filter != 'All' and 'region' in drill_df.columns:
        drill_df = drill_df[drill_df['region'] == region_filter]
        
    if segment_filter != 'All' and 'segment' in drill_df.columns:
        drill_df = drill_df[drill_df['segment'] == segment_filter]
        
    if ship_mode_filter != 'All' and 'ship_mode' in drill_df.columns:
        drill_df = drill_df[drill_df['ship_mode'] == ship_mode_filter]
        
    if discount_band != 'All' and 'discount' in drill_df.columns:
        if discount_band == '0-10%':
            drill_df = drill_df[drill_df['discount'] <= 0.1]
        elif discount_band == '10-20%':
            drill_df = drill_df[(drill_df['discount'] > 0.1) & (drill_df['discount'] <= 0.2)]
        elif discount_band == '20-30%':
            drill_df = drill_df[(drill_df['discount'] > 0.2) & (drill_df['discount'] <= 0.3)]
        elif discount_band == '30%+':
            drill_df = drill_df[drill_df['discount'] > 0.3]
            
    if profit_threshold != 'All' and 'profit' in drill_df.columns:
        if profit_threshold == 'Loss-making only':
            drill_df = drill_df[drill_df['profit'] < 0]
        elif profit_threshold == 'Profitable only':
            drill_df = drill_df[drill_df['profit'] > 0]
        elif profit_threshold == 'Break-even':
            drill_df = drill_df[drill_df['profit'] == 0]
    
    if drill_df.empty:
        st.warning("No data matches the selected drill filters.")
        return
    
    # Loss driver analysis
    st.subheader("📊 Loss Driver Analysis")
    blackholes = analyze_profitability_blackholes(drill_df, top_n=20)
    
    if not blackholes.empty:
        # Pareto chart
        pareto_fig = get_pareto_chart(blackholes)
        st.plotly_chart(pareto_fig, use_container_width=True)
        st.markdown("*Use this chart to identify the 20% of items driving 80% of losses*")
        
        # Loss details table
        st.subheader("📋 High-Loss Items Details")
        st.dataframe(
            blackholes.head(10),
            use_container_width=True,
            hide_index=True
        )
        
        # Auto-generated insights
        st.markdown("### 🔍 Generated Insights")
        
        # Find specific problem combinations  
        problem_combos = []
        for _, row in blackholes.head(5).iterrows():
            item_data = drill_df[
                (drill_df['product_name'].str.contains(row['Item'].split(' ')[0], na=False)) |
                (drill_df['sub_category'] == row['Item'])
            ]
            if not item_data.empty:
                avg_discount = item_data['discount'].mean()
                main_region = item_data['region'].mode().iloc[0] if not item_data['region'].mode().empty else 'Unknown'
                problem_combos.append(f"• **{main_region} + {row['Item']}** with {avg_discount:.1%} avg discount = **${abs(row['Loss']):,.0f}** loss")
        
        for combo in problem_combos[:3]:
            st.markdown(combo)
        
        # Export functionality
        if st.button("📥 Export Loss Black Holes"):
            csv_data = blackholes.to_csv(index=False)
            st.download_button(
                label="Download loss_blackholes.csv",
                data=csv_data,
                file_name="loss_blackholes.csv",
                mime="text/csv"
            )
    else:
        st.info("No significant loss drivers found in the selected data.")

def customer_value_tab(filtered):
    """Customer Value RFM segmentation tab"""
    guard_empty(filtered)
    
    st.header("👥 Customer Value (RFM Segmentation)")
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
        st.subheader("🏆 Customer Tier Distribution")
        tier_counts = customer_segments['Tier'].value_counts()
        
        for tier in ['VIP', 'Loyal', 'Promising', 'At-Risk']:
            if tier in tier_counts.index:
                count = tier_counts[tier]
                percentage = (count / len(customer_segments)) * 100
                st.metric(f"{tier} Customers", f"{count:,} ({percentage:.1f}%)")
    
    with col2:
        st.subheader("💰 Revenue by Tier")
        
        # Calculate revenue and profit by tier
        tier_metrics = customer_segments.groupby('Tier').agg({
            'Monetary': 'sum',
            'customer_id': 'count'
        }).round(2)
        tier_metrics.columns = ['Total_Revenue', 'Customer_Count']
        
        for tier in tier_metrics.index:
            revenue = tier_metrics.loc[tier, 'Total_Revenue']
            st.metric(f"{tier} Revenue", f"${revenue:,.0f}")
    
    # RFM Cohort Heatmap
    st.subheader("🗺️ RFM Cohort Analysis")
    
    # Create cohort by Region
    cohort_data = customer_segments.merge(
        filtered[['customer_id', 'region']].drop_duplicates(),
        on='customer_id'
    )
    
    cohort_matrix = cohort_data.groupby(['Tier', 'region']).size().reset_index(name='Count')
    cohort_pivot = cohort_matrix.pivot(index='Tier', columns='region', values='Count').fillna(0)
    
    if not cohort_pivot.empty:
        import plotly.express as px
        fig = px.imshow(
            cohort_pivot.values,
            x=cohort_pivot.columns,
            y=cohort_pivot.index,
            color_continuous_scale='Blues',
            title="Customer Tiers by Region Heatmap"
        )
        fig.update_layout(
            xaxis_title="Region",
            yaxis_title="Customer Tier"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("*Use this heatmap to identify regional customer value patterns*")
    
    # RFM insights
    st.markdown("### 🔍 Customer Value Insights")
    insights = get_rfm_insights(customer_segments)
    for insight in insights:
        st.markdown(f"• {insight}")
    
    # Export RFM data
    if st.button("📥 Export RFM Customer Data"):
        export_data = customer_segments[['Customer.ID', 'R_Score', 'F_Score', 'M_Score', 'Tier']]
        csv_data = export_data.to_csv(index=False)
        st.download_button(
            label="Download rfm_customers.csv",
            data=csv_data,
            file_name="rfm_customers.csv",
            mime="text/csv"
        )

def discount_elasticity_tab(filtered):
    """Discount elasticity and pricing analysis tab"""
    guard_empty(filtered)
    
    st.header("💰 Discount Elasticity & Pricing")
    st.markdown("*Analyze the relationship between discounts and profitability to optimize pricing strategies*")
    
    # Elasticity analysis
    elasticity_data = analyze_discount_elasticity(filtered)
    
    if elasticity_data.empty:
        st.warning("Unable to perform elasticity analysis with current data.")
        return
    
    # Elasticity explorer charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💹 Discount vs Profit per Order")
        import plotly.express as px
        
        profit_fig = px.scatter(
            elasticity_data,
            x='Discount_Band_Mid',
            y='Median_Profit_Per_Order',
            size='Order_Count',
            title="Profit Elasticity by Discount Band",
            labels={
                'Discount_Band_Mid': 'Discount Rate',
                'Median_Profit_Per_Order': 'Median Profit per Order ($)'
            }
        )
        profit_fig.update_traces(marker=dict(color='red', opacity=0.7))
        profit_fig.update_layout(showlegend=False)
        st.plotly_chart(profit_fig, use_container_width=True)
        st.markdown("*Identify the discount sweet spot for maximum profitability*")
    
    with col2:
        st.subheader("📊 Discount vs Sales per Order")
        
        sales_fig = px.scatter(
            elasticity_data,
            x='Discount_Band_Mid',
            y='Median_Sales_Per_Order',
            size='Order_Count',
            title="Sales Volume by Discount Band",
            labels={
                'Discount_Band_Mid': 'Discount Rate',
                'Median_Sales_Per_Order': 'Median Sales per Order ($)'
            }
        )
        sales_fig.update_traces(marker=dict(color='blue', opacity=0.7))
        sales_fig.update_layout(showlegend=False)
        st.plotly_chart(sales_fig, use_container_width=True)
        st.markdown("*Track how discounts drive sales volume*")
    
    # What-if simulation
    st.subheader("🎯 What-If Discount Simulation")
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
    
    simulation_result = simulate_profit_impact(filtered, max_discount)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Total Profit", f"${simulation_result['current_profit']:,.0f}")
    with col2:
        st.metric("Simulated Profit", f"${simulation_result['simulated_profit']:,.0f}")
    with col3:
        impact = simulation_result['simulated_profit'] - simulation_result['current_profit']
        st.metric("Profit Impact", f"${impact:,.0f}", delta=f"{impact:,.0f}")
    
    # Elasticity insights
    st.markdown("### 🔍 Pricing Insights")
    
    # Find optimal discount range
    optimal_discount = elasticity_data.loc[elasticity_data['Median_Profit_Per_Order'].idxmax()]
    high_volume_discount = elasticity_data.loc[elasticity_data['Order_Count'].idxmax()]
    
    insights = [
        f"• **Optimal profitability** achieved at **{optimal_discount['Discount_Band']}** discount range",
        f"• **Highest order volume** occurs at **{high_volume_discount['Discount_Band']}** discount range",
        f"• Capping discounts at **{max_discount:.0%}** would result in **${abs(impact):,.0f}** profit {'gain' if impact > 0 else 'loss'}**"
    ]
    
    for insight in insights:
        st.markdown(insight)
    
    # Export elasticity data
    if st.button("📥 Export Elasticity Analysis"):
        csv_data = elasticity_data.to_csv(index=False)
        st.download_button(
            label="Download elasticity_summary.csv",
            data=csv_data,
            file_name="elasticity_summary.csv",
            mime="text/csv"
        )

def operations_crosssell_tab(filtered):
    """Operations and cross-sell analysis tab"""
    guard_empty(filtered)
    
    st.header("🔄 Operations & Cross-Sell")
    st.markdown("*Operational efficiency analysis and cross-selling opportunities*")
    
    # Shipping efficiency analysis
    st.subheader("🚚 Shipping Efficiency Analysis")
    
    shipping_analysis = filtered.groupby(['ship_mode', 'region']).agg({
        'profit': 'mean',
        'order_id': 'count',
        'sales': 'mean'
    }).round(2)
    shipping_analysis.columns = ['Avg_Profit', 'Order_Count', 'Avg_Sales']
    
    # Create shipping heatmap
    shipping_pivot = shipping_analysis.reset_index().pivot(
        index='ship_mode', 
        columns='region', 
        values='Avg_Profit'
    )
    
    if not shipping_pivot.empty:
        import plotly.express as px
        fig = px.imshow(
            shipping_pivot.values,
            x=shipping_pivot.columns,
            y=shipping_pivot.index,
            color_continuous_scale='RdYlGn',
            title="Average Profit by Ship Mode × Region"
        )
        fig.update_layout(
            xaxis_title="Region",
            yaxis_title="Shipping Mode"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("*Identify the most profitable shipping mode-region combinations*")
    
    # Market Basket Analysis
    st.subheader("🛒 Market Basket Analysis")
    
    with st.expander("⚙️ Advanced Settings"):
        min_support = st.slider("Minimum Support", 0.01, 0.05, 0.02, 0.01)
        max_len = st.slider("Maximum Rule Length", 2, 5, 2)
    
    try:
        basket_rules = perform_market_basket_analysis(filtered, min_support=min_support, max_len=max_len)
        
        if not basket_rules.empty:
            st.markdown("#### 🎯 Top Association Rules by Lift")
            
            # Display top rules
            top_rules = basket_rules.nlargest(10, 'lift')[
                ['antecedents', 'consequents', 'support', 'confidence', 'lift']
            ].round(3)
            
            st.dataframe(top_rules, use_container_width=True, hide_index=True)
            st.markdown("*Higher lift values indicate stronger associations between products*")
            
            # Export basket rules
            if st.button("📥 Export Basket Rules"):
                csv_data = basket_rules.to_csv(index=False)
                st.download_button(
                    label="Download basket_rules.csv",
                    data=csv_data,
                    file_name="basket_rules.csv",
                    mime="text/csv"
                )
        else:
            st.info(f"No association rules found with minimum support of {min_support}. Try lowering the support threshold.")
    
    except Exception as e:
        st.warning(f"Market basket analysis failed: {str(e)}. This may be due to insufficient data or missing dependencies.")
    
    # Segment Profitability Matrix
    st.subheader("📊 Segment Profitability Matrix")
    
    profitability_matrix = get_segment_profitability_matrix(filtered)
    
    if not profitability_matrix.empty:
        # Create heatmap for segment profitability
        matrix_pivot = profitability_matrix.pivot_table(
            index='segment',
            columns=['category', 'region'],
            values='Profit_Margin',
            aggfunc='mean'
        ).fillna(0)
        
        if not matrix_pivot.empty:
            import plotly.express as px
            fig = px.imshow(
                matrix_pivot.values,
                x=[f"{cat}-{reg}" for cat, reg in matrix_pivot.columns],
                y=matrix_pivot.index,
                color_continuous_scale='RdYlGn',
                title="Profit Margin % by Segment × Category × Region"
            )
            fig.update_layout(
                xaxis_title="Category - Region",
                yaxis_title="Customer Segment",
                xaxis=dict(tickangle=45)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("*Identify the most profitable segment-category-region combinations*")
    
    # Optional ML Profitability Prediction
    st.subheader("🤖 ML-Powered Profitability Prediction")
    
    with st.expander("⚙️ ML Model Settings"):
        enable_ml = st.checkbox("Enable Profitability Prediction", value=False)
        model_type = st.selectbox("Model Type", ["Random Forest", "Logistic Regression"])
    
    if enable_ml:
        try:
            with st.spinner("Training profitability prediction model..."):
                model_results = train_profitability_model(filtered, model_type=model_type.replace(" ", "_").lower())
            
            if model_results:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Model Accuracy", f"{model_results['accuracy']:.3f}")
                    st.metric("Precision-Recall AUC", f"{model_results['pr_auc']:.3f}")
                
                with col2:
                    st.markdown("#### 🔍 Top Profitability Drivers")
                    insights = get_model_insights(model_results)
                    for insight in insights:
                        st.markdown(f"• {insight}")
                
                # Feature importance chart
                if 'feature_importance' in model_results and not model_results['feature_importance'].empty:
                    import plotly.express as px
                    fig = px.bar(
                        model_results['feature_importance'].head(10),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 10 Profitability Drivers"
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Export predictions
                if st.button("📥 Export Profitability Predictions") and 'predictions' in model_results:
                    csv_data = model_results['predictions'].to_csv(index=False)
                    st.download_button(
                        label="Download profitability_predictions.csv",
                        data=csv_data,
                        file_name="profitability_predictions.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("Unable to train profitability model with current data.")
                
        except Exception as e:
            st.warning(f"ML model training failed: {str(e)}. This feature requires additional dependencies.")
    
    # Optional Choropleth Map
    st.subheader("🗺️ Geographic Profit Analysis")
    
    show_map = st.checkbox("Show State-Level Choropleth Map", value=False)
    
    if show_map:
        try:
            choropleth_fig = plot_state_map(filtered, value_col='profit', title='Profit by State')
            if choropleth_fig:
                st.plotly_chart(choropleth_fig, use_container_width=True)
                st.markdown("*Hover over states to see detailed metrics including sales and average discount*")
            else:
                st.info("Unable to create choropleth map with current data.")
        except Exception as e:
            st.warning(f"Choropleth map failed to load: {str(e)}")

if __name__ == "__main__":
    main()
