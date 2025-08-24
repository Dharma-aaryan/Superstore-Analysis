# Superstore Insights Dashboard

## Overview

This is a Streamlit-based business analytics dashboard designed for superstore data analysis. The application provides executive-ready insights through multiple analytical modules including profitability analysis, customer segmentation, discount elasticity, and predictive modeling. The dashboard is built with a modular architecture that separates data processing, analysis, and visualization concerns into distinct utility modules.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web-based dashboard interface
- **Layout**: Wide layout with expandable sidebar for global filters
- **Navigation**: Tab-based interface for different analytical views (Executive Summary, Profitability Black Holes, Customer Value, Discount Elasticity)
- **Visualization**: Plotly for interactive charts and graphs
- **State Management**: Streamlit's built-in session state with sidebar filters that persist across tabs

### Backend Architecture
- **Data Processing**: Pandas-based ETL pipeline with automatic data type inference and cleaning
- **Performance Optimization**: Streamlit caching decorators (@st.cache_data for data operations, @st.cache_resource for ML models)
- **Modular Design**: Utility modules organized by functionality:
  - `data.py`: Data loading and cleaning operations
  - `profit.py`: Profitability analysis and black hole identification
  - `rfm.py`: Customer segmentation using RFM (Recency, Frequency, Monetary) analysis
  - `elasticity.py`: Discount elasticity analysis with binning approach
  - `basket.py`: Market basket analysis using Apriori algorithm
  - `model.py`: Machine learning models for profitability prediction
  - `viz.py`: Visualization components and KPI calculations

### Data Storage Solutions
- **Input**: CSV file upload with fallback to default superstore.csv
- **Processing**: In-memory pandas DataFrames with caching
- **Output**: Downloadable CSV exports for analysis results
- **No persistent database**: Application operates on uploaded/default data files

### Authentication and Authorization
- **Access Control**: None implemented (public dashboard)
- **Security Model**: File-based data access only

## External Dependencies

### Python Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualization library
- **scikit-learn**: Machine learning algorithms (RandomForest, LogisticRegression)
- **mlxtend**: Market basket analysis (Apriori algorithm, association rules)
- **scipy**: Statistical analysis and computations

### Data Requirements
- **Primary**: superstore.csv file or user-uploaded CSV
- **Schema Expected**: Order.Date, Sales, Profit, Customer.ID, Product.Name, Category, Region, Discount, Quantity, Ship.Mode columns
- **Format**: CSV with datetime parsing for date columns

### Third-party Services
- **None**: Application runs locally without external API dependencies
- **Deployment**: Designed for Replit or local Streamlit deployment

### Machine Learning Components
- **Models**: RandomForest and LogisticRegression for profitability prediction
- **Features**: Sales, Discount, Quantity, Shipping.Cost plus categorical encodings
- **Target**: Binary classification (profitable vs unprofitable orders)
- **Preprocessing**: LabelEncoder for categorical variables, StandardScaler for numerical features