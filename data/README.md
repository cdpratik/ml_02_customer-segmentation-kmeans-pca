# Data Directory

## 📁 Directory Structure
data/
├── raw/ # Original, immutable data
├── processed/ # Cleaned and transformed datasets
└── external/ # Additional reference data (if any)

text


## 📥 Dataset Download Instructions

### Online Retail II Dataset

**Source:** https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

**Steps:**
1. Visit the UCI Machine Learning Repository link above
2. Click on "Data Folder"
3. Download `online_retail_II.xlsx`
4. Place the file in `data/raw/` directory

**File Name:** `online_retail_II.xlsx`  
**Expected Location:** `data/raw/online_retail_II.xlsx`

> ⚠️ **Note:** Raw data files are NOT tracked in Git (see `.gitignore`)

## 📊 Dataset Information

### Overview
- **Domain:** E-commerce / Retail
- **Type:** Transactional data
- **Period:** December 2009 - December 2011
- **Records:** ~525,000 transactions
- **Customers:** ~4,300 unique customers
- **Products:** ~3,600 unique products

### Schema

| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `InvoiceNo` | String | 6-digit invoice number. Prefix 'C' indicates cancellation | 536365, C536379 |
| `StockCode` | String | 5-digit product code | 85123A |
| `Description` | String | Product name | WHITE HANGING HEART T-LIGHT HOLDER |
| `Quantity` | Integer | Number of items per transaction | 6 |
| `InvoiceDate` | Datetime | Transaction date and time | 2010-12-01 08:26:00 |
| `UnitPrice` | Float | Price per unit in GBP (£) | 2.55 |
| `CustomerID` | Float | 5-digit customer identifier | 17850.0 |
| `Country` | String | Customer's country of residence | United Kingdom |

### Data Characteristics

**Positive Indicators:**
- ✅ Real-world transactional data
- ✅ Rich temporal information
- ✅ Multiple customer behaviors captured
- ✅ International customer base

**Known Issues:**
- ⚠️ Missing CustomerID values (~25% of records)
- ⚠️ Cancelled transactions (indicated by 'C' prefix)
- ⚠️ Negative quantities (returns/corrections)
- ⚠️ Zero/negative prices (special cases)
- ⚠️ Duplicate records
- ⚠️ Non-product entries (postage, fees, etc.)

## 🔄 Data Processing Pipeline
Raw Data (data/raw/)
↓
Data Validation
↓
Data Cleaning
↓
Feature Engineering
↓
Processed Data (data/processed/)

text


### Processed Data Files (Generated)

| File | Description |
|------|-------------|
| `cleaned_transactions.csv` | Cleaned transactional data |
| `customer_features.csv` | Customer-level aggregated features |
| `rfm_data.csv` | RFM (Recency, Frequency, Monetary) features |
| `final_features.csv` | Final feature set for clustering |

## 📏 Data Quality Checks

Before modeling, we validate:
- [ ] Schema compliance
- [ ] Date range validity
- [ ] Numeric field ranges
- [ ] Missing value patterns
- [ ] Duplicate detection
- [ ] Business logic validation

## 🔒 Data Privacy & Ethics

- Dataset is publicly available and anonymized
- No personally identifiable information (PII)
- CustomerID is a pseudonymous identifier
- Complies with academic research usage guidelines

## 📚 References

- **Citation:** Chen, D., Sain, S.L. and Guo, K., 2012. Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. Journal of Database Marketing & Customer Strategy Management, 19(3), pp.197-208.

- **Original Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)

## ⚠️ Important Notes

1. **DO NOT** commit raw data files to Git
2. **DO** document any data preprocessing steps
3. **DO** keep track of data versions
4. **DO** validate data after each transformation

---

**Last Updated:** [Add Date]