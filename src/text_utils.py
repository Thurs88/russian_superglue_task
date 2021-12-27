import re


def prepare_data(df):
    """
    Simple text preprocessing function
    :param df:
    :return:
    """
    # strip dash but keep a space
    df['premise'] = df['premise'].str.replace('-', ' ')
    df['hypothesis'] = df['hypothesis'].str.replace('-', ' ')
    # lower case the data
    df['premise'] = df['premise'].apply(lambda x: x.lower())
    df['hypothesis'] = df['hypothesis'].apply(lambda x: x.lower())

    # remove excess spaces near punctuation
    df['premise'] = df['premise'].apply(lambda x: re.sub(r'\s([?.!"](?:\s|$))', r'\1', x))
    df['hypothesis'] = df['hypothesis'].apply(lambda x: re.sub(r'\s([?.!"](?:\s|$))', r'\1', x))

    # remove excess white spaces
    df['premise'] = df['premise'].apply(lambda x: " ".join(x.split()))
    df['hypothesis'] = df['hypothesis'].apply(lambda x: " ".join(x.split()))

    return df
