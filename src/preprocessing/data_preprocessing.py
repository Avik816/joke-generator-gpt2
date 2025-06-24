import polars
import re
import html


def checking_for_url(text):
    url_pattern = re.compile(r'https?://|www\.|\.com|\.net|\.org|\.io|\.gov|\.in|\.me|\.co', re.IGNORECASE)

    return bool(url_pattern.search(str(text)))

"""def deleting_quotes(text):
    return (
        text.replace('“', '"').replace('”', '"').
        replace('‘', "'").replace('’', "'")
    )

def removing_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002700-\U000027BF"  # Dingbats
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\U00002500-\U00002BEF"  # CJK (some emoji/symbol overlap)
        "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r'', str(text))

def checking_for_html_entities(text):
    html_entity_pattern = re.compile(r"&(gt|lt|amp|quot|apos);", re.IGNORECASE)

    return bool(html_entity_pattern.search(str(text)))

def checking_for_square_brackets(text):
    return bool(re.search(r"\[|\]", str(text)))

def clean_text(text):
    text = str(text).strip()
    text = deleting_quotes(text)
    text = removing_emojis(text)
    text = re.sub(r'\s+', ' ', text)
    return text"""

# Applying the custom functions on the dataset
def clean_dataset(dataset):

    # Checking and removing rows with NaN and Null values
    print('\nNull Count ...\n', dataset.null_count())
    dataset = dataset.drop_nulls().drop_nulls()
    print(f'After Null and NaN removal, Dataset Shape: {dataset.shape}')

    # Checking and removing duplicated values
    print(f'\nDataset is Duplicated ?: {dataset.is_duplicated().any()}')
    dataset = dataset.unique()
    print(f'After duplicate value removal, Dataset Shape: {dataset.shape}')
    
    # Filtering out URLs
    dataset = dataset.filter(~dataset.map_rows(checking_for_url).to_series())

    
    


dataset = polars.read_csv('datasets/dad-jokes.csv')
print(f'Original Shape: {dataset.shape}')

clean_dataset(dataset)