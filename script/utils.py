import ast
# Utility function to convert raw data in CSV
def convert_cols(x):
    if isinstance(x, int) or isinstance(x, float) or isinstance(x, list):
        return x
    try:
        x = ast.literal_eval(x)
    finally:
        return x

def convert_cols_mod(x):
    if isinstance(x, int) or isinstance(x, float):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        try:
            x = ast.literal_eval(x)
        finally:
            if isinstance(x, list):
                return tuple(x)
            return x


# Relevance is an integer, the higher the better
def rank2relevance(df, top_k, col_rank):
    return top_k + 1 - df[col_rank].values.ravel()
