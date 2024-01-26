from functools import partial
from typing import List, Tuple, Callable, Dict, Optional
import pandas as pd
import os, itertools, json, glob, hashlib
from .attribute import AttributeTree

concat_conversation = lambda conversation: ('||').join(conversation)
deconcat_conversation = lambda conversation: conversation.split('||')

def generate_hash(conversation):
    conversation = concat_conversation(conversation)
    # Convert the conversation to a string representation if it's not already
    conv_str = str(conversation)
    # Create a hash of the conversation
    return hashlib.sha256(conv_str.encode()).hexdigest()

def expand_unannotated(unannotated: pd.DataFrame):
    unannotated_combinations = []
    # Iterate through each row and column
    for attribute in unannotated.columns:    
        for index, row in unannotated.iterrows():   
            if not row[attribute]: 
                unannotated_combinations.append((index[0], index[1], attribute))
    return unannotated_combinations

def create_temp_anno_record(N):
    return {i : [0,0] for i in range(N)}

# Parse whatever dataset of conversations into list of conversations
# This one works if your conversatons is stored in a folder of json files
def parse_conversations(folder_path: str = './data/conversation/') -> List[str]:
    conversation_files = glob.glob(f'{folder_path}conversation_*.json')
    conversations = []
    for conversation_file in conversation_files:
        with open(conversation_file, 'r') as f:
            conversation = json.load(f)
            conversations.append(conversation)
    return conversations


def remove_duplicate_in_hash_dict(hash_dict, storage_ids):
    # List to keep track of hash_ids to be removed
    hash_ids_to_remove = []

    # Check HashID against duplications
    for hash_id in hash_dict:
        if hash_id in storage_ids:
            # Add hash_id to the list for removal
            hash_ids_to_remove.append(hash_id)

    # Remove the identified hash_ids from hash_dict
    for hash_id in hash_ids_to_remove:
        hash_dict.pop(hash_id)
    return hash_dict

parse_new_conversations = partial(parse_conversations, folder_path='./data/conversation/')


class PoeBaseDataset:

    def __init__(self, conversations: pd.DataFrame, 
                 annotations: pd.DataFrame, 
                 annotator_info: str, 
                 attribute_info: Dict,
                 store_dir: str):
        # Storage directory
        self.store_dir = store_dir
        # Conversations
        self.conversations = conversations
        # Annotations
        self.annotations = annotations
        # Annotator Info
        self.annotator_info = annotator_info
        # Attribute Info
        self.attribute_info = attribute_info
        # Prepare for annotation -- only on pair-wise comparison not done yet
        unannotated = self.get_unannotated_pairs_attributes()
        self._prepare_anno()

    def get_unannotated_pairs_attributes(self) -> pd.DataFrame:
        # get all the pairs of conversations
        hash_id_pairs = list(itertools.combinations(list(self.conversations['hash_id']), 2))
        # get all the attributes
        attributes = list(self.attribute_info.keys())
        # Create a MultiIndex
        multi_index = pd.MultiIndex.from_tuples(hash_id_pairs, names=['hash_id_a', 'hash_id_b'])
        # Initialize an empty DataFrame with this MultiIndex and attributes as columns
        self.anno_info = pd.DataFrame(index=multi_index, columns=attributes, dtype=bool, data=False)

        # get all the pairs of conversations that have not been annotated for each attribute, default value to False
        # loop through current annotations to get the annotated pairs
        for idx, row in self.annotations.iterrows():
            hash_id_a = row['hash_id_a']
            hash_id_b = row['hash_id_b']
            # get the pair of conversations
            pair = (hash_id_a, hash_id_b)
            # get the attribute
            attribute = row['attribute']
            
            # mark the pair of conversations as annotated for the attribute
            self.anno_info.loc[pair, attribute] = True



        # get the un-annotated pairs of conversations || those entry with False value
        self.unannotated = self.anno_info[~self.anno_info.any(axis=1)]
        # self.unannotated = self.anno_info[self.anno_info.isfalse().any(axis=1)]
        return self.unannotated
        

    @classmethod
    def load_conversations(cls, store_dir: str) -> pd.DataFrame:
        # Load the conversations as pandas dataframe || Dataframe: hash_id, conversation
        conversation_file_path = f'{store_dir}conversations.csv'
        if not os.path.exists(conversation_file_path):
            conversations = pd.DataFrame(columns=['hash_id', 'conversation'])
        else:
            conversations = pd.read_csv(conversation_file_path)
        return conversations
    
    @classmethod
    def save_conversations(cls, conversations: pd.DataFrame, store_dir: str) -> None:
        # Save the conversations as pandas dataframe || Dataframe: hash_id, conversation
        conversation_file_path = f'{store_dir}conversations.csv'
        conversations.to_csv(f'{store_dir}conversations.csv', index=False)

    @classmethod
    def load_annotations(cls, store_dir: str, annotator_info: str) -> pd.DataFrame:
        # Load the annotations as pandas dataframe || (hash_id, hash_id), preference
        anno_file_path = f'{store_dir}annotation_{annotator_info}.csv'
        if not os.path.exists(anno_file_path):
            annotations = pd.DataFrame(columns=['hash_id_a', 'hash_id_b', 'preference', 'attribute'])
        else:
            annotations = pd.read_csv(anno_file_path)
        return annotations
    
    def save_annotations(self) -> None:
        # Save the annotations as pandas dataframe || (hash_id, hash_id), preference
        anno_file_path = f'{self.store_dir}annotation_{self.annotator_info}.csv'
        self.annotations.to_csv(anno_file_path, index=False)

    @classmethod
    def load(cls, store_dir: str, annotator_info: str, attribute_info: Dict) -> 'POEDataset':
        # Load the conversations
        conversations = cls.load_conversations(store_dir)
        # Load the annotations
        annotations = cls.load_annotations(store_dir, annotator_info)
        # Create the dataset
        return cls(conversations, annotations, annotator_info, attribute_info, store_dir)
    
    @classmethod
    def load_storage_ids(cls, store_dir: str = './data/annotation/') -> List[str]:
        # Load the conversations as pandas dataframe || Dataframe: hash_id, conversation
        conversations = cls.load_conversations(store_dir)
        if len(conversations) > 0:
            return conversations['hash_id'].tolist()
        else:
            return []
    
    @classmethod
    def merge_conversations(cls, store_dir: str, hash_dict: Dict[str, List[str]], return_df: bool = False) -> Optional[pd.DataFrame]:
        # Load the conversations as pandas dataframe || Dataframe: hash_id, conversation
        conversations = cls.load_conversations(store_dir)
        # Convert the hash_dict to a dataframe
        new_conversations = pd.DataFrame(hash_dict.items(), columns=['hash_id', 'conversation'])
        # Merge the new conversations into the storage
        conversations = pd.concat([conversations, new_conversations], ignore_index=True)
        # Save the new conversations
        cls.save_conversations(conversations, store_dir)
        if return_df:
            return conversations

    @classmethod
    def make(cls, parse_new_conversations: Callable,
             attribute_tree: AttributeTree, annotator_info: str, 
             store_dir: str = './data/annotation/'):
        
        # Load in-storage ids
        storage_ids: List[str] = cls.load_storage_ids(store_dir)

        # Hash the new-conversations (potentially not in storage)
        raw_conversations = parse_new_conversations()
        hash_dict = {generate_hash(conversation): concat_conversation(conversation) for conversation in raw_conversations}
        
        # Check HashID against duplications
        hash_dict = remove_duplicate_in_hash_dict(hash_dict, storage_ids)

        # Merge the new conversations into the storage
        cls.merge_conversations(store_dir, hash_dict)

        # get the attribute leaf nodes 
        attribute_leaf_nodes = attribute_tree.get_leaf_nodes()
        attribute_info = {node.value: node.subjective_score for node in attribute_leaf_nodes}

        # Load the dataset
        return cls.load(store_dir, annotator_info, attribute_info)
    
    def retrieve_conversation(self, hash_id):
        conversation = self.conversations[self.conversations['hash_id']==hash_id]['conversation'].iloc[0]
        return deconcat_conversation(conversation)
    
    # So, During Annotation, we evaluate each attribute independently 
    # -- loop through attribute, subloop through pairs of conversation

    def _prepare_anno(self) -> None:
        self.update_annotation_summary()
        self.unannotated_combinations = expand_unannotated(self.unannotated)
        self.temp_anno_record = create_temp_anno_record(len(self.unannotated_combinations))

    def _idx_to_hash_id_pair(self, idx: int) -> Tuple[int, int]:
        hash_id_a, hash_id_b, attribute = self.unannotated_combinations[idx]
        return hash_id_a, hash_id_b

    # def _pair_idx_to_idx(self, pair_idx: Tuple[int, int]) -> int:
        # return self.indices.index(pair_idx)

    def __len__(self) -> int:
        return len(self.unannotated_combinations)
    
    def annotate(self, idx: int, preference:  Tuple[int, int]) -> None:
        # annotation goes into temporay buffer to record one-time selection (change is possible)
        self.temp_anno_record[idx] = preference

    def _cache_anno(self) -> None:
        # Cache the annotation
        for idx, preference in self.temp_anno_record.items():
            if (preference[0] + preference[1]) == 0:
                continue
            hash_id_a, hash_id_b, attribute = self.unannotated_combinations[idx]
            self.annotations.loc[len(self.annotations)] = [hash_id_a, hash_id_b, preference, attribute]
        # Save the annotation
        self.save_annotations()
        # Update the unannotated pairs
        self.unannotated = self.get_unannotated_pairs_attributes()
        # Prepare for annotation
        self._prepare_anno()

    def save(self) -> None:
        # Cache the annotation
        self._cache_anno()

    def update_annotation_summary(self):
        # Update the annotation summary
        # self.annotation_summary = self.annotations.groupby(['hash_id_a', 'hash_id_b']).agg({'preference': 'sum'}).reset_index()
        # self.annotation_summary['preference'] = self.annotation_summary['preference'].apply(lambda x: 1 if x > 0 else -1)
        # self.annotation_summary = self.annotation_summary.rename(columns={'preference': 'preference_sum'})
        pass


class POEDataset(PoeBaseDataset):
    def __getitem__(self, idx):
        if idx < len(self.unannotated_combinations):
            hash_id_a, hash_id_b, attribute = self.unannotated_combinations[idx]
            return self.retrieve_conversation(hash_id_a), self.retrieve_conversation(hash_id_b), attribute
        else:
            raise IndexError("Index out of range")
    def __iter__(self):
        self._iter_idx = 0
        return self
    def __next__(self):
        if self._iter_idx >= len(self):
            raise StopIteration
        self._iter_idx += 1
        hash_id_a, hash_id_b, attribute = self.unannotated_combinations[self._iter_idx]
        return self.retrieve_conversation(hash_id_a), self.retrieve_conversation(hash_id_b), attribute
    

def parse_conversation_into_name_and_messages(conversation):
    name, message = conversation.split(':')
    return name, message