import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

class FM_Preprocessing:
    def __init__(self, df, target_col='target', num_epochs=10):
        self.df = df
        self.target_col = target_col
        self.num_epochs = num_epochs
        self.X_tensor, self.y_tensor, self.c_values_tensor, self.user_feature_tensor, self.item_feature_tensor, self.all_item_ids, self.num_features = self.prepare_data()
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The df parameter should be a pandas DataFrame.")
        
        if target_col not in df.columns:
            raise ValueError(f"The target column {target_col} is not in the DataFrame.")

    def generate_not_purchased_data(df):
        customer_frequency, product_frequency 컬럼 추가
        df['customer_frequency'] = df.groupby('AUTH_CUSTOMER_ID')['AUTH_CUSTOMER_ID'].transform('count')
        df['product_frequency'] = df.groupby('PRODUCT_CODE')['PRODUCT_CODE'].transform('count')
    
        
        unique_customers = df['AUTH_CUSTOMER_ID'].unique()
        unique_products = df['PRODUCT_CODE'].unique()
    
        
        not_purchased_products_list = []
    
        
        for customer in unique_customers:
            customer_frequency = df[df['AUTH_CUSTOMER_ID'] == customer]['customer_frequency'].iloc[0]
            purchased_products = df[df['AUTH_CUSTOMER_ID'] == customer]['PRODUCT_CODE'].unique()

            customer_birth_category = df[df['AUTH_CUSTOMER_ID'] == customer]['Birth_Category'].iloc[0]
            customer_gender_category = df[df['AUTH_CUSTOMER_ID'] == customer]['gender_category'].iloc[0]
        
            not_purchased_products = [product for product in unique_products if product not in purchased_products]
        
            not_purchased_products_data = [{'AUTH_CUSTOMER_ID': customer, 
                                        'PRODUCT_CODE': product,
                                        'Birth_Category': customer_birth_category,
                                        'gender_category': customer_gender_category,
                                        'customer_frequency': customer_frequency,
                                        'product_frequency': df[df['PRODUCT_CODE'] == product]['product_frequency'].iloc[0]} 
                                       for product in not_purchased_products]
        
            not_purchased_products_list.extend(not_purchased_products_data)
    
    
        not_purchased_df = pd.DataFrame(not_purchased_products_list)
        not_purchased_df['target'] = 0
    
        return not_purchased_df

    def prepare_data(self):
        X = self.df.drop(columns=[self.target_col, 'PRODUCT_CODE', 'C', 'AUTH_CUSTOMER_ID'])
        y = self.df[self.target_col]
        c = self.df['C']
        
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1)

        c_values_tensor = torch.tensor(c, dtype=torch.float32)
        c_values_tensor = torch.where(c_values_tensor < 1, c_values_tensor * 100, c_values_tensor)

        unique_user_df = self.df.drop_duplicates(subset=['AUTH_CUSTOMER_ID']).sort_values('AUTH_CUSTOMER_ID')
        user_features_df = unique_user_df[['Birth_Category', 'gender_category']]
        user_feature_tensor = torch.tensor(pd.get_dummies(user_features_df).values, dtype=torch.float32)

        unique_item_df = self.df.drop_duplicates(subset=['PRODUCT_CODE']).sort_values('PRODUCT_CODE')
        item_features_df = unique_item_df.filter(like='DEPTH')
        item_feature_tensor = torch.tensor(item_features_df.values, dtype=torch.float32)

        all_item_ids = list(self.df.PRODUCT_CODE.unique())

        num_features = X.shape[1]
        
        return X_tensor, y_tensor, c_values_tensor, user_feature_tensor, item_feature_tensor, all_item_ids, num_features

if __name__ == '__main__':
    
    try:
        df = pd.read_csv('your_data.csv')
        preprocess = FM_Preprocessing(df)
        not_purchased_df = preprocess.generate_not_purchased_data()
        # ...
    except Exception as e:
        print(f"An error occurred: {e}")
        



class FactorizationMachine(nn.Module):
    def __init__(self, num_features, num_factors, lr=0.01, weight_decay=0.01):
        super(FactorizationMachine, self).__init__()
        self.num_features = num_features
        self.num_factors = num_factors
        self.w = nn.Parameter(torch.randn(num_features))
        self.v = nn.Parameter(torch.randn(num_features, num_factors))
        self.optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        #self.loss_func = nn.BCEWithLogitsLoss()
        #self.loss_func =  nn.MSELoss()
        self.weight_decay = weight_decay

    def forward(self, x):
        linear_terms = torch.matmul(x, self.w)
        interactions = 0.5 * torch.sum(
            torch.matmul(x, self.v) ** 2 - torch.matmul(x ** 2, self.v ** 2),
            dim=1,
            keepdim=True
        )
        return linear_terms + interactions.squeeze()

    def loss(self, y_pred, y_true, c_values):
        mse = (y_pred - y_true.float()) ** 2
        weighted_mse = c_values * mse
        l2_reg = torch.norm(self.w)**2 + torch.norm(self.v)**2  # L2 regularization
        return torch.mean(weighted_mse) + self.weight_decay * l2_reg

    def train_step(self, x, y, c_values):
        self.optimizer.zero_grad()
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y, c_values)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def recommend_top_n_items(self, user_features, all_item_features, all_item_ids, top_n=5):
        combined_features = torch.cat([user_features.expand(all_item_features.shape[0], -1), all_item_features], dim=1)
        with torch.no_grad():
            scores = self.forward(combined_features)
        sorted_indices = torch.argsort(scores, descending=True)[:top_n]
        return [all_item_ids[i] for i in sorted_indices]

    def recommend_top_n_items_for_all_users(self, user_features_list, all_item_features, all_item_ids, top_n=5):
        recommendations = {}
        for i, user_features in enumerate(user_features_list):
            user_id = i  # can replace with actual user ID if I have
            top_n_items = self.recommend_top_n_items(user_features, all_item_features, all_item_ids, top_n)
            recommendations[user_id] = top_n_items
        return recommendations



# for epoch in range(num_epochs):
#     loss = model.train_step(X_tensor, y_tensor,c_values_tensor)
#     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}')

# # Make recommendations
# recommendations = model.recommend_top_n_items_for_all_users(user_feature_tensor, item_feature_tensor, all_item_ids, top_n=5)
# print("User-wise top 5 recommended items:", recommendations)
