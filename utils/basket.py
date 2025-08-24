import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def perform_market_basket_analysis(df, min_support=0.02, max_len=2):
    """Perform market basket analysis using Apriori algorithm"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Import mlxtend for association rules
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
        
        # Create transaction dataset (orders with products)
        transactions = df.groupby('order_id')['product_name'].apply(list).tolist()
        
        # Filter out very long transactions (might be data quality issues)
        transactions = [t for t in transactions if len(t) <= 20]
        
        if not transactions:
            return pd.DataFrame()
        
        # Create binary matrix for market basket analysis  
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        basket_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Find frequent itemsets
        frequent_itemsets = apriori(
            basket_df, 
            min_support=min_support, 
            use_colnames=True,
            max_len=max_len
        )
        
        if frequent_itemsets.empty:
            return pd.DataFrame()
        
        # Generate association rules
        rules = association_rules(
            frequent_itemsets, 
            metric="lift", 
            min_threshold=1.0,
            num_itemsets=len(frequent_itemsets)
        )
        
        if rules.empty:
            return pd.DataFrame()
        
        # Clean up the rules dataframe
        rules_clean = rules.copy()
        rules_clean['antecedents'] = rules_clean['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_clean['consequents'] = rules_clean['consequents'].apply(lambda x: ', '.join(list(x)))
        
        # Select relevant columns
        rules_final = rules_clean[[
            'antecedents', 'consequents', 'antecedent support', 'consequent support',
            'support', 'confidence', 'lift', 'conviction'
        ]].copy()
        
        rules_final.columns = [
            'antecedents', 'consequents', 'antecedent_support', 'consequent_support',
            'support', 'confidence', 'lift', 'conviction'
        ]
        
        # Filter for meaningful rules (lift > 1.1, confidence > 0.3)
        rules_final = rules_final[
            (rules_final['lift'] > 1.1) & 
            (rules_final['confidence'] > 0.3)
        ].sort_values('lift', ascending=False)
        
        return rules_final
        
    except ImportError:
        st.warning("mlxtend library not available. Market basket analysis disabled.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error performing market basket analysis: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def get_segment_profitability_matrix(df):
    """Create segment profitability matrix"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Create profitability matrix by multiple dimensions
        matrix_data = df.groupby(['segment', 'category', 'region']).agg({
            'sales': 'sum',
            'profit': 'sum',
            'order_id': 'nunique',
            'customer_id': 'nunique'
        }).reset_index()
        
        # Calculate key metrics
        matrix_data['Profit_Margin'] = (matrix_data['profit'] / matrix_data['sales']) * 100
        matrix_data['Avg_Order_Value'] = matrix_data['sales'] / matrix_data['order_id']
        matrix_data['Sales_Per_Customer'] = matrix_data['sales'] / matrix_data['customer_id']
        
        # Filter for segments with meaningful data
        matrix_data = matrix_data[
            (matrix_data['order_id'] >= 5) & 
            (matrix_data['customer_id'] >= 3)
        ]
        
        return matrix_data.round(2)
        
    except Exception as e:
        st.error(f"Error creating segment profitability matrix: {str(e)}")
        return pd.DataFrame()

def get_cross_sell_opportunities(basket_rules):
    """Identify cross-selling opportunities from association rules"""
    if basket_rules.empty:
        return []
    
    opportunities = []
    
    try:
        # Focus on high-lift, high-confidence rules
        top_rules = basket_rules[
            (basket_rules['lift'] > 1.5) & 
            (basket_rules['confidence'] > 0.4)
        ].head(10)
        
        for _, rule in top_rules.iterrows():
            opportunity = {
                'primary_product': rule['antecedents'],
                'recommended_product': rule['consequents'],
                'confidence': rule['confidence'],
                'lift': rule['lift'],
                'support': rule['support'],
                'recommendation': f"Customers who buy {rule['antecedents']} are {rule['lift']:.1f}x more likely to buy {rule['consequents']} (confidence: {rule['confidence']:.1%})"
            }
            opportunities.append(opportunity)
        
        return opportunities
        
    except Exception as e:
        st.error(f"Error identifying cross-sell opportunities: {str(e)}")
        return []

@st.cache_data
def analyze_shipping_profitability(df):
    """Analyze profitability by shipping mode and region"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Shipping analysis
        shipping_analysis = df.groupby(['ship_mode', 'region']).agg({
            'profit': ['sum', 'mean'],
            'sales': ['sum', 'mean'],
            'shipping_cost': 'mean',
            'order_id': 'nunique'
        }).reset_index()
        
        # Flatten column names
        shipping_analysis.columns = [
            'Ship_Mode', 'Region', 'Total_Profit', 'Avg_Profit',
            'Total_Sales', 'Avg_Sales', 'Avg_Shipping_Cost', 'Order_Count'
        ]
        
        # Calculate metrics
        shipping_analysis['Profit_Margin'] = (
            shipping_analysis['Total_Profit'] / shipping_analysis['Total_Sales']
        ) * 100
        
        shipping_analysis['Profit_Per_Order'] = (
            shipping_analysis['Total_Profit'] / shipping_analysis['Order_Count']
        )
        
        # Calculate shipping cost efficiency
        shipping_analysis['Shipping_Cost_Ratio'] = (
            shipping_analysis['Avg_Shipping_Cost'] / shipping_analysis['Avg_Sales']
        ) * 100
        
        return shipping_analysis.round(2)
        
    except Exception as e:
        st.error(f"Error analyzing shipping profitability: {str(e)}")
        return pd.DataFrame()

def get_operational_insights(df):
    """Generate operational insights from the data"""
    if df.empty:
        return []
    
    insights = []
    
    try:
        # Shipping mode analysis
        if 'ship_mode' in df.columns:
            shipping_profit = df.groupby('ship_mode')['profit'].mean().sort_values(ascending=False)
            best_shipping = shipping_profit.index[0]
            worst_shipping = shipping_profit.index[-1]
            
            profit_diff = shipping_profit.iloc[0] - shipping_profit.iloc[-1]
            insights.append(
                f"**{best_shipping}** shipping is **${profit_diff:.0f}** more profitable per order than **{worst_shipping}**"
            )
        
        # Regional performance
        if 'region' in df.columns:
            region_margins = df.groupby('region').agg({
                'profit': 'sum',
                'sales': 'sum'
            })
            region_margins['Margin'] = (region_margins['profit'] / region_margins['sales']) * 100
            
            best_region = region_margins['Margin'].idxmax()
            best_margin = region_margins.loc[best_region, 'Margin']
            
            insights.append(
                f"**{best_region}** region shows highest profit margin at **{best_margin:.1f}%**"
            )
        
        # Category concentration
        if 'category' in df.columns:
            category_revenue = df.groupby('category')['sales'].sum()
            total_revenue = category_revenue.sum()
            top_category_share = (category_revenue.max() / total_revenue) * 100
            top_category = category_revenue.idxmax()
            
            insights.append(
                f"**{top_category}** dominates with **{top_category_share:.1f}%** of total sales, indicating strong category focus"
            )
        
        return insights[:3]
        
    except Exception as e:
        st.error(f"Error generating operational insights: {str(e)}")
        return []

@st.cache_data  
def get_customer_segment_behavior(df):
    """Analyze behavior patterns by customer segment"""
    if df.empty or 'segment' not in df.columns:
        return pd.DataFrame()
    
    try:
        segment_behavior = df.groupby('segment').agg({
            'sales': ['sum', 'mean'],
            'profit': ['sum', 'mean'],
            'discount': 'mean',
            'quantity': 'mean',
            'order_id': 'nunique',
            'customer_id': 'nunique'
        }).reset_index()
        
        # Flatten columns
        segment_behavior.columns = [
            'segment', 'Total_Sales', 'Avg_Order_Value', 'Total_Profit',
            'Avg_Profit_Per_Order', 'Avg_Discount', 'Avg_Quantity', 
            'Total_Orders', 'Total_Customers'
        ]
        
        # Calculate derived metrics
        segment_behavior['Orders_Per_Customer'] = (
            segment_behavior['Total_Orders'] / segment_behavior['Total_Customers']
        )
        
        segment_behavior['Profit_Margin'] = (
            segment_behavior['Total_Profit'] / segment_behavior['Total_Sales']
        ) * 100
        
        segment_behavior['Revenue_Per_Customer'] = (
            segment_behavior['Total_Sales'] / segment_behavior['Total_Customers']
        )
        
        return segment_behavior.round(2)
        
    except Exception as e:
        st.error(f"Error analyzing customer segment behavior: {str(e)}")
        return pd.DataFrame()
