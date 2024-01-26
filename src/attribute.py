from .utils import TreeNode

# Build a Tree of Conversation Attributes
def build_conversation_attribute_tree_samual1():
    root = TreeNode('* GoodConversations_S')
    child1 = TreeNode('Grammatical_Accuracy_S')
    child2 = TreeNode('Socialiguistic_Proficiency_S')
    child3 = TreeNode('Contextual_Awareness_S')
    child4 = TreeNode('Persona_Performance_S')
    child5 = TreeNode('Communication_Strategies_S')
    child6 = TreeNode('DifficultyPortrayal_S')

    root.add_child(child1)
    root.add_child(child2)
    root.add_child(child3)
    root.add_child(child4)
    root.add_child(child5)
    root.add_child(child6)

    child1.add_child(TreeNode('LanguageUse_S', color='red'))
    child1.add_child(TreeNode('Grammar_O', color='red'))
    child1.add_child(TreeNode('Spelling_O', color='red'))

    child2.add_child(TreeNode('Slang_S', color='red'))
    child2.add_child(TreeNode('DemographicAutheticity_S', color='red'))

    child3.add_child(TreeNode('Retention_S', color='red'))
    child3.add_child(TreeNode('TopicRelevance_S', color='red'))

    child4.add_child(TreeNode('CustomerBackstory_S', color='red'))
    child4.add_child(TreeNode('CustomerHobby_S', color='red'))

    child41 = TreeNode('CustomerConcern_S')
    child4.add_child(child41)
    child41.add_child(TreeNode('FinancialConcern_S', color='red'))
    child41.add_child(TreeNode('HealthConcern_S', color='red'))
    child41.add_child(TreeNode('InsuranceNeed_S', color='red'))

    child61 = TreeNode('SkepticalNavigator_S', color='red')
    child61.add_child(TreeNode('Demanding_and_High_Expectation_S', color='red'))
    child61.add_child(TreeNode('Extreme_Price_Sensitivity_S', color='red'))
    child61.add_child(TreeNode('High_Skepticism_about_Insurance_Benefits_S', color='red'))
    child61.add_child(TreeNode('Detail-Oriented_and_Meticulous_S', color='red'))
    child61.add_child(TreeNode('Security_and_Privacy-Conscious_S', color='red'))
    child61.add_child(TreeNode('Past_Nagative_Experience_with_Insurance_S', color='red'))
    
    child61.add_child(TreeNode('Show_Irational_Distrust_O', color='red'))
    child61.add_child(TreeNode('Prejudice_S', color='red')) # this one will not easily get synthetic data on
    child61.add_child(TreeNode('Denial_of_agent_credibility_O', color='red'))


    child6.add_child(TreeNode('SkepticalNavigator_S', color='red'))
    child6.add_child(TreeNode('ConversionCriterion_S', color='red'))

    # break the ICE | let the customer feels like they've been listened to 
    # Communication Strategies
    child5.add_child(TreeNode('SmallTalkEffectiveness_S', color='red'))
    child5.add_child(TreeNode('Empathy_S', color='red'))
    child5.add_child(TreeNode('ActiveListening_S', color='red'))
    child5.add_child(TreeNode('Overcoming_Communication_Breakdown_S', color='red'))
    child5.add_child(TreeNode('AskClarifyingQuestion_to_address_ambiguity_S', color='red'))

    return root

# With Samual's input, I prompt GPT4 to give me a simpler ones:
# Help me simplify the attributes such that:
# 1. No more than 8 leaf node
# 2. No more than 3 layers
# 3. leaf node should be objective attribute that is easy to evaluate & compare
# step 0 into decomposition - with few-shot example ;>
def build_conversation_attribute_tree_gpt1():
    root = TreeNode('* ConversationQuality_S')

    # Primary Categories
    clarity = TreeNode('Clarity_S')
    engagement = TreeNode('Engagement_S')
    relevance = TreeNode('Relevance_S')

    root.add_child(clarity)
    root.add_child(engagement)
    root.add_child(relevance)

    # Clarity Subcategories (Leaf Nodes)
    clarity.add_child(TreeNode('Grammar_Accuracy_O', color='red'))
    clarity.add_child(TreeNode('Clear_Expression_O', color='red'))

    # Engagement Subcategories (Leaf Nodes)
    engagement.add_child(TreeNode('Active_Listening_O', color='red'))
    engagement.add_child(TreeNode('Empathy_Expression_O', color='red'))

    # Relevance Subcategories (Leaf Nodes)
    relevance.add_child(TreeNode('Contextual_Appropriateness_O', color='red'))
    relevance.add_child(TreeNode('Topical_Focus_O', color='red'))

    return root

# Samual iteration 1
def build_conversation_attribute_tree():
    root = TreeNode('* ConversationQuality_S')

    # Primary Categories
    language = TreeNode('Language_Use_S')
    persona = TreeNode('PersonaAuthenticity_S')
    relevance = TreeNode('Relevance_S')
    coherence = TreeNode('Coherence_S')

    root.add_child(language)
    root.add_child(persona)
    root.add_child(relevance)
    root.add_child(coherence)

    # Language Subcategories (Leaf Nodes)
    language.add_child(TreeNode('Grammar_Accuracy_O', color='red'))
    language.add_child(TreeNode('Slang_O', color='red'))
    language.add_child(TreeNode('Naturalness_S', color='red'))

    # Persona Subcategories (Leaf Nodes)
    persona.add_child(TreeNode('CustomerSmallTalk_S', color='red'))
    persona.add_child(TreeNode('SkepticalNavigator_O', color='red')) # demanding, high expectation, high skepticisim about insurance benefits, detail-oriented and meticulous, security and privacy conscious, irational distrust, prejudice, denial of agent credibiliy

    # Relevance Subcategories (Leaf Nodes)
    relevance.add_child(TreeNode('Contextual_Consistency_O', color='red')) # Do not double ask something, which is like you have gold-fish memory.
    relevance.add_child(TreeNode('Topic_Relevance_O', color='red')) # unless there is resonable justification on topic change, switch topic is bad for this.

    #`Coherence Subcategories (Leaf Nodes) | i+1 sentence coherence with i sentence
    coherence.add_child(TreeNode('Coherent_Utterance_O', color='red')) # is i+1 related to i? or completely separated and inhuman?
    return root

# Build a Tree of Personality Attributes
def build_personality_attribute_tree_alice():
    root = TreeNode('* HardToSell_S')
    child1 = TreeNode('FutureOriented_S')
    child2 = TreeNode('RiskTolerance_S')
    child3 = TreeNode('Conscientiousness_S')
    child4 = TreeNode('Neuroticism_S')
    
    root.add_child(child1)
    root.add_child(child2)
    root.add_child(child3)
    root.add_child(child4)

    child2.add_child(TreeNode('Anxiety_O', color='red'))
    child2.add_child(TreeNode('Cautiousness_O', color='red'))

    child4.add_child(TreeNode('Impetience_O', color='red'))
    child4.add_child(TreeNode('Rudeness_O', color='red'))

    return root

# Alice's prompt -> GPT4 revise version
def build_personality_attribute_tree():
    root = TreeNode('* PersonalityTraits_S')

    # Primary Categories
    openness = TreeNode('Openness_S')
    conscientiousness = TreeNode('Conscientiousness_S')
    extraversion = TreeNode('Extraversion_S')
    agreeableness = TreeNode('Agreeableness_S')
    neuroticism = TreeNode('Neuroticism_S')

    root.add_child(openness)
    root.add_child(conscientiousness)
    root.add_child(extraversion)
    root.add_child(agreeableness)
    root.add_child(neuroticism)

    # Openness Leaf Nodes
    openness.add_child(TreeNode('Creativity_O', color='red'))
    openness.add_child(TreeNode('Curiosity_O', color='red'))

    # Conscientiousness Leaf Nodes
    conscientiousness.add_child(TreeNode('Efficiency_O', color='red'))
    conscientiousness.add_child(TreeNode('Organization_O', color='red'))

    # Extraversion Leaf Nodes
    extraversion.add_child(TreeNode('Sociability_O', color='red'))
    extraversion.add_child(TreeNode('Assertiveness_O', color='red'))

    # Agreeableness Leaf Nodes
    agreeableness.add_child(TreeNode('Compassion_O', color='red'))
    agreeableness.add_child(TreeNode('Cooperation_O', color='red'))

    # Neuroticism Leaf Nodes
    neuroticism.add_child(TreeNode('Anxiety_O', color='red'))
    neuroticism.add_child(TreeNode('MoodSwings_O', color='red'))

    return root

def build_conversation_attribute_tree_test():
    root = TreeNode('* ConversationQuality_S')
    child1 = TreeNode('Consistency_S')
    root.add_child(child1)
    return root

def build_personality_attribute_tree_test():
    root = TreeNode('* HardToSell_S')
    child1 = TreeNode('Crazy_S')
    root.add_child(child1)
    return root

# AttributeTree wrapps conversation & personality attribute trees
from dataclasses import dataclass
@dataclass
class AttributeTree:
    conversation_tree: TreeNode
    personality_tree: TreeNode
    name: str = 'AttributeTree_AICustomer'

    @classmethod
    def make(cls):
        return AttributeTree(
            conversation_tree=build_conversation_attribute_tree_test(),
            personality_tree=build_personality_attribute_tree_test()
        )
    
    def get_leaf_nodes(self):
        return self.conversation_tree.get_leaf_nodes() + self.personality_tree.get_leaf_nodes()






