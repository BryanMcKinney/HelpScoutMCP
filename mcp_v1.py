import pandas as pd
import sqlite3
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class QueryResult:
    """Structure to hold query results and metadata"""
    data: Any
    query_type: str
    sql_query: Optional[str] = None
    error: Optional[str] = None
    explanation: Optional[str] = None

class NaturalLanguageProcessor:
    """Processes natural language queries and converts them to SQL or pandas operations"""
    
    def __init__(self):
        # Common patterns for different query types
        self.patterns = {
            'count': [
                r'how many', r'count', r'number of', r'total.*records',
                r'how much', r'quantity'
            ],
            'filter': [
                r'where', r'with', r'having', r'contains?', r'equals?',
                r'greater than', r'less than', r'between', r'in the range'
            ],
            'aggregate': [
                r'average', r'mean', r'sum', r'total', r'maximum', r'minimum',
                r'max', r'min', r'avg', r'top', r'correlation'
            ],
            'group': [
                r'group by', r'grouped by', r'by category', r'breakdown',
                r'per', r'each', r'for every'
            ],
            'sort': [
                r'sort', r'order', r'rank', r'top', r'bottom', r'highest', r'lowest'
            ]
        }
        
        # Common operators
        self.operators = {
            'greater than': '>',
            'less than': '<',
            'equals': '=',
            'equal to': '=',
            'not equal': '!=',
            'contains': 'LIKE'
        }
    
    def identify_intent(self, query: str) -> List[str]:
        """Identify the intent(s) of the natural language query"""
        query_lower = query.lower()
        intents = []
        
        for intent, patterns in self.patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                intents.append(intent)
        
        return intents if intents else ['general']
    
    def extract_column_references(self, query: str, columns: List[str]) -> List[str]:
        """Extract column names referenced in the query"""
        query_lower = query.lower()
        found_columns = []
        
        for col in columns:
            col_variations = [
                col.lower(),
                col.lower().replace('_', ' '),
                col.lower().replace('_', ''),
                ' '.join(word.capitalize() for word in col.split('_'))
            ]
            
            if any(variation in query_lower for variation in col_variations):
                found_columns.append(col)
        
        return found_columns
    
    def extract_values(self, query: str) -> Dict[str, Any]:
        """Extract numeric values and string literals from query"""
        values = {}
        
        # Extract numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        if numbers:
            values['numbers'] = [float(n) for n in numbers]
        
        # Extract quoted strings
        quoted_strings = re.findall(r'["\']([^"\']+)["\']', query)
        if quoted_strings:
            values['strings'] = quoted_strings
        
        return values

class DatasetQuerier:
    """Main class for handling dataset queries"""
    
    def __init__(self, data_source: Any, source_type: str = 'pandas'):
        """
        Initialize with a data source
        
        Args:
            data_source: pandas DataFrame, SQLite connection, or file path
            source_type: 'pandas', 'sqlite', or 'csv'
        """
        self.source_type = source_type
        self.nlp = NaturalLanguageProcessor()
        
        if source_type == 'pandas':
            self.df = data_source
            self.columns = list(self.df.columns)
        elif source_type == 'csv':
            self.df = pd.read_csv(data_source)
            self.columns = list(self.df.columns)
        elif source_type == 'sqlite':
            self.conn = data_source
            self.columns = self._get_sqlite_columns()
        
    def _get_sqlite_columns(self) -> List[str]:
        """Get column names from SQLite database"""
        cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        if tables:
            table_name = tables[0][0]
            cursor = self.conn.execute(f"PRAGMA table_info({table_name});")
            return [row[1] for row in cursor.fetchall()]
        return []
    
    def query(self, natural_query: str) -> QueryResult:
        """Process a natural language query and return results"""
        try:
            # Identify query intent
            intents = self.nlp.identify_intent(natural_query)
            
            # Extract relevant columns
            relevant_columns = self.nlp.extract_column_references(natural_query, self.columns)
            
            # Extract values
            values = self.nlp.extract_values(natural_query)
            
            # Build and execute query based on intent
            if 'count' in intents:
                return self._handle_count_query(natural_query, relevant_columns, values)
            elif 'aggregate' in intents:
                return self._handle_aggregate_query(natural_query, relevant_columns, values)
            elif 'filter' in intents:
                return self._handle_filter_query(natural_query, relevant_columns, values)
            elif 'group' in intents:
                return self._handle_group_query(natural_query, relevant_columns, values)
            else:
                return self._handle_general_query(natural_query, relevant_columns)
                
        except Exception as e:
            return QueryResult(
                data=None,
                query_type='error',
                error=str(e)
            )
    
    def _handle_count_query(self, query: str, columns: List[str], values: Dict) -> QueryResult:
        """Handle counting queries"""
        if self.source_type == 'pandas':
            if not columns:
                count = len(self.df)
                explanation = f"Total number of records: {count}"
            else:
                # Count non-null values in specific columns
                col = columns[0]
                count = self.df[col].count()
                explanation = f"Number of non-null values in '{col}': {count}"
            
            return QueryResult(
                data=count,
                query_type='count',
                explanation=explanation
            )
    
    def _handle_aggregate_query(self, query: str, columns: List[str], values: Dict) -> QueryResult:
        """Handle aggregation queries (sum, mean, max, min, top)"""
        if not columns:
            return QueryResult(data=None, query_type='aggregate', 
                             error="No numeric columns found in query")
        
        query_lower = query.lower()
        col = columns[0]
        #print(columns)
        
        if self.source_type == 'pandas':
            if 'average' in query_lower or 'mean' in query_lower:
                result = self.df[col].mean()
                operation = 'average'
            elif 'sum' in query_lower or 'total' in query_lower:
                result = self.df[col].sum()
                operation = 'sum'
            elif 'max' in query_lower or 'maximum' in query_lower:
                result = self.df[col].max()
                operation = 'maximum'
            elif 'min' in query_lower or 'minimum' in query_lower:
                result = self.df[col].min()
                operation = 'minimum'
            elif 'top' in query_lower:
                result = self.df[col].head(10)
                operation = 'top'
            elif 'correlation' in query_lower:
                corr_value = self.df[columns[0]].corr(self.df[columns[1]])
                if corr_value <= 0:
                    result = "no correlation can be determined"
                if corr_value > .75:
                    result = "high correlation between variables"
                if corr_value < .75 and corr_value > 0:
                    result = "some correlation between variables"
                operation = 'correlation'
            else:
                result = self.df[col].describe()
                operation = 'summary statistics'
            
            return QueryResult(
                data=result,
                query_type='aggregate',
                explanation=f"Calculated {operation} for column '{col}'"
            )
    
    def _handle_filter_query(self, query: str, columns: List[str], values: Dict) -> QueryResult:
        """Handle filtering queries"""
        if not columns:
            return QueryResult(data=None, query_type='filter',
                             error="No columns identified for filtering")
        
        if self.source_type == 'pandas':
            df_filtered = self.df.copy()
            
            # Simple filtering based on values found in query
            if 'numbers' in values and len(values['numbers']) > 0:
                col = columns[0]
                value = values['numbers'][0]
                
                if 'greater than' in query.lower():
                    df_filtered = df_filtered[df_filtered[col] > value]
                elif 'less than' in query.lower():
                    df_filtered = df_filtered[df_filtered[col] < value]
                else:
                    df_filtered = df_filtered[df_filtered[col] == value]
            
            return QueryResult(
                data=df_filtered,
                query_type='filter',
                explanation=f"Filtered data based on conditions in '{columns[0]}'"
            )
    
    def _handle_group_query(self, query: str, columns: List[str], values: Dict) -> QueryResult:
        """Handle grouping queries"""
        if not columns:
            return QueryResult(data=None, query_type='group',
                             error="No columns identified for grouping")
        
        if self.source_type == 'pandas':
            group_col = columns[0]
            
            # Simple grouping with count
            if 'count' in query.lower():
                result = self.df.groupby(group_col).size()
            else:
                # Default to count if no specific aggregation mentioned
                result = self.df.groupby(group_col).size()
            
            return QueryResult(
                data=result,
                query_type='group',
                explanation=f"Grouped data by '{group_col}'"
            )
    
    def _handle_general_query(self, query: str, columns: List[str]) -> QueryResult:
        """Handle general queries - show relevant columns or basic info"""
        if self.source_type == 'pandas':
            if columns:
                # Show specific columns
                result = self.df[columns]
                explanation = f"Showing columns: {', '.join(columns)}"
            else:
                # Show basic dataset info
                result = {
                    'shape': self.df.shape,
                    'columns': list(self.df.columns),
                    'dtypes': dict(self.df.dtypes),
                    'sample': self.df.head().to_dict()
                }
                explanation = "Basic dataset information"
            
            return QueryResult(
                data=result,
                query_type='general',
                explanation=explanation
            )

class ConversationalInterface:
    """Main interface for the conversational prototype"""
    
    def __init__(self, dataset_path: str = None, dataset: pd.DataFrame = None):
        """Initialize with either a file path or pandas DataFrame"""
        if dataset is not None:
            self.querier = DatasetQuerier(dataset, 'pandas')
        elif dataset_path:
            if dataset_path.endswith('.csv'):
                self.querier = DatasetQuerier(dataset_path, 'csv')
            else:
                raise ValueError("Unsupported file format. Currently supports CSV files.")
        else:
            raise ValueError("Must provide either dataset_path or dataset")
        
        self.conversation_history = []
    
    def ask(self, question: str) -> str:
        """Ask a natural language question about the dataset"""
        # Store the question
        self.conversation_history.append(('user', question))
        
        # Process the query
        result = self.querier.query(question)
        
        # Format the response
        response = self._format_response(result)
        
        # Store the response
        self.conversation_history.append(('assistant', response))
        
        return response
    
    def _format_response(self, result: QueryResult) -> str:
        """Format the query result into a human-readable response"""
        if result.error:
            return f"Sorry, I encountered an error: {result.error}"
        
        response = []
        
        if result.explanation:
            response.append(result.explanation)
        
        if result.query_type == 'count':
            response.append(f"Result: {result.data}")
        elif result.query_type == 'aggregate':
            if isinstance(result.data, (int, float)):
                response.append(f"Result: {result.data:.2f}")
            else:
                response.append(f"Result:\n{result.data}")
        elif result.query_type == 'filter':
            if hasattr(result.data, 'shape'):
                response.append(f"Found {len(result.data)} matching records")
                if len(result.data) <= 10:
                    response.append(f"\n{result.data.to_string()}")
                else:
                    response.append(f"\nShowing first 5 records:\n{result.data.head().to_string()}")
        elif result.query_type == 'group':
            response.append(f"Results:\n{result.data}")
        elif result.query_type == 'general':
            if isinstance(result.data, dict):
                response.append(f"Dataset Info: {json.dumps(result.data, indent=2, default=str)}")
            else:
                response.append(f"Results:\n{result.data}")
        
        return '\n'.join(response)
    
    def show_columns(self) -> str:
        """Show available columns in the dataset"""
        return f"Available columns: {', '.join(self.querier.columns)}"
    
    def show_sample(self, n: int = 5) -> str:
        """Show sample data"""
        if self.querier.source_type == 'pandas':
            return f"Sample data:\n{self.querier.df.head(n).to_string()}"

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for demonstration
    sample_data = pd.read_csv('saas_customer_data.csv')
    
    # Data Cleaning steps of sample_data
    sample_data.columns = [col.lower().replace(' ', '_') for col in sample_data.columns]
    # Harcoded clean up based on current data set, need to add in more dynamic cleaning 
    # for future state
    sample_data['new_freq'] = sample_data['payment_frequency'].case_when(
        [
            (sample_data['payment_frequency'] == 'Monthly', 12),
            (sample_data['payment_frequency'] == 'Quarterly', 4),
            (sample_data['payment_frequency'] == 'Annually', 1)
        ]
    )
    sample_data['billings'] = sample_data['billings'].str.replace('$', '')
    sample_data['billings'] = sample_data['billings'].str.replace(',', '').astype(float)
    sample_data['contacts'] = sample_data['contacts'].str.replace(',', '').astype(float)
    sample_data['yearly_revenue'] = sample_data['billings'] * sample_data['new_freq']
    sample_data = sample_data.sort_values(by='yearly_revenue', ascending=False)
    
   
    # Initialize the conversational interface
    mcp = ConversationalInterface(dataset=sample_data)
    
    # Example conversations
    print("=== Minimal Conversational Prototype Demo ===\n")
    
    print("Available columns:")
    print(mcp.show_columns())
    print("\n" + "="*50 + "\n")
    
    # Test different types of queries
    test_queries = [
        "How many distinct customers do we have?",
        "What are the top 10 company_name based on yearly_revenue?",
        "Is there a correlation between contacts and workflows?",
        "What is the average yearly revenue by customer?"
    ]
    
    for query in test_queries:
        print(f"Q: {query}")
        print(f"A: {mcp.ask(query)}")
        print("\n" + "-"*30 + "\n")
