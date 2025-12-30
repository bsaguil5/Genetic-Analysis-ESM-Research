"""
ESM-based Protein Classifier Architecture

This module defines the classifier model that takes ESM embeddings as input
and performs binary classification on protein sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import config

# This method is used to take the embeddings that we get after passing our sequences through ESM2
# and then create a summary vector with 480 numbers for each protein
class AttentionPooling(nn.Module):
    """
    Attention-based pooling for sequence representations.
    
    Learns to weight different positions in the sequence for classification.
    Often works better than simple mean/max pooling for proteins.
    """
    # is constructed when AttentionPooling is called????????????
    # input dimension is the dimension that we start out with
    # hidden dimension is the optional one, the one where we use to squash 480 to 128 to
    def __init__(self, input_dim: int, hidden_dim: int = None):
        """
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for attention computation (default from config)
        """
        super().__init__() # calls the parent class (nn.Module) so that pyTorch recognizes this as a neural network layer

        # Use config default if not specified
        hidden_dim = hidden_dim or config.ATTENTION_HIDDEN_DIM
 #?   
        self.attention = nn.Sequential( # 480 -> 128 -> tanh -> 1
            nn.Linear(input_dim, hidden_dim), # squash from 480 to 128 (truncating it down helps the neural network have more capacity to learn complex patterns in dimensionality
            # if it was 480 -> 1, which is more direct, it cna only learn simple linear combinations, which say nothing about dimensionality, really. but going from 480->128->1 
            # gives it the ability to possibly learn complex patterns and how dimensions much have to do with one another.
            # dim X and dim Y matter together or dim Z matters only if dim W is low type of thing.
            # you don't really get that when you just have 480 straight to 1 i think you lose that

            nn.Tanh(), # unlinearize (standard practice) to make the values in the vector go from -1 to 1. 
            nn.Linear(hidden_dim, 1) # 128 dimensions into 1 (would be a score)
        ) # ^^Builds the machinery for our forward function to use and truncate the 480 dim vector into a summary vector, eventually
        

    # embeddings: shape (batch_size, seq_len, embed_dim) -> like (16, 200, 480) = 16 proteins, 200 positions, 480 dimensional vector for each position    
  #?# mask: shape (batch_size, seq_len) says which positions (where?) are real vs padding
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: # this is the method that runs when you actually use AttentionPooling in ESMClassifier
        
        # Compute attention weights
        attention_weights = self.attention(embeddings)  # (batch, seq_len, 1) # compute attention weights
        # input (16, 200, 480) -> output (16,200,1)
        # 16 proteins with 200 scores
        # 16 separate arrays of 200 scores
        """  Example for one protein:
            Position 1: [0.5, -0.3, ..., 0.2] → attention network →
            [0.73]
            Position 2: [0.1, 0.7, ..., 0.9]  → attention network →
            [0.52]
            Position 3: [-0.2, 0.3, ..., 0.4] → attention network →
            [0.91]
            ...
HDJSAKDSHAJKGLAKJSDKGLASKDJ
HDJSAKDSHAJKGLAKJSDKGLASKDJ
HDJSAKDSHAJKGLAKJSDKGLASKDJ
HDJSAKDSHAJKGLAKJSDKGLASKDJ
HDJSAKDSHAJKGLAKJSDKGLASKDJ
  [12] [12]
            squeeze (-1)
        """
        attention_weights = attention_weights.squeeze(-1)  # (batch, seq_len)w
        # 16 arrays of 200 scores but more simply
        # an array of size 200 with just the scores
# what does .squeeze(-1) do?
# it removes the rapper [] so that it just becomes a simple list of 200 scores :)
        # ^^ how important is each position?
        # takes the position embeddings, runs through the attention network to get the SCORE.
        # so in the 16 proteins example where u have 200 positions, you get 200 scores

        # Mask out padding positions
        if mask is not None:
            # Convert mask to boolean if needed
            if mask.dtype != torch.bool:
                mask = mask.bool()
# make sure that the mask is bool True/False values so that torch can work with it

            attention_weights = attention_weights.masked_fill(~mask, float('-inf'))
        # ~mask means NOT, flips the boolean, will say True for False and False for True
        # mask_fill wherever ~mask is true (wherever we want to mask) because it's marked as F
        # example: positions 1 2 3 4 5 6 7
        #          mask      T T T T T F F
        #  attention scores  # # # # # # #   where the #s are the scores
        # after we do the mask_fill:
                             # # # # # -inf -inf]


        # how does softmax give us normalized weights and what are normalized weights
        # and why do we need them?
        # converts raw scores into probabilities together when added equal 1
        # it also makes the -inf 0 because e^-inf is 0 -> padding positions get 0 weight
        attention_weights = F.softmax(attention_weights, dim=1)  # (batch, seq_len)
        
        # attention_weights.unsqueeze(-1) adds a dimension
        # before we had (batch, seq_len) (16,100)
        # now we'll have (batch, seq_len, 1) (16, 100, 1)
        # so that way we can multiply with embeddings (16, 200, 480)
        # (16 , 200 , 480) * (16, 200, 1) = (16, 200, 1) multiplies each 480 dim embedding by the score
        # which is what we we're trying to do, we have one 480 dim vector with scores applied
        # then when we add all 480 dim embeddings of the 200 weighted position vectors we get one 480 dim vector per protein
        pooled = torch.sum(embeddings * attention_weights.unsqueeze(-1), dim=1)  # (batch, embed_dim)
        # for one protein
        # embeddings (200 positions, 480 numbers)
        """  Position 1: [0.5, -0.3, 0.8, ..., 0.2]  ← 480 numbers
            Position 2: [0.1, 0.7, -0.4, ..., 0.9]  ← 480 numbers
            ...
            Position 200: [0.4, -0.1, 0.7, ..., 0.6] ← 480 numbers"""
        
        # each positions 480 number array gets scaled by the single score
        """  Then torch.sum(..., dim=1) adds all 200 weighted
            position vectors:
            Summary = [0.12 + 0.017 + ... + 0.084,
                        -0.072 + 0.119 + ... + -0.021,
                        ...]
                    = [final_480_numbers]  ← ONE 480-number array
            per protein"""


        # returns the summary vector
        # (16,480)
        return pooled

# takes 480 dimensional vectors from ESM2 (batch, seq_len, 480)
# if embeddings are too large this method compresses them
# uses AttentionPooling
# has the classification head (neural network)
# outputs logits: raw prediction scores
class ESMClassifier(nn.Module):

    
    def __init__(self, 
                 esm_embed_dim: int, # required: 480 dimensions
                 hidden_dim: int = None, # optional: 128
                 pooling_strategy: str = None, # optional: "attention", "mean", "max"
                 dropout: float = None, # optional: dropout rain for regularization (prevents overfitting)
#                 
                 
                 num_classes: int = 1): # number of output choices

        super().__init__()
        
        # Use config defaults if not specified
        hidden_dim = hidden_dim or config.CLASSIFIER_HIDDEN_DIM
        # config value might be 256
        pooling_strategy = pooling_strategy or config.POOLING_STRATEGY
        # config.POOLING_STRAGY eg "attention"

       
        dropout = dropout or config.CLASSIFIER_DROPOUT
        # e.g. .3 for 30% dropout    


        self.esm_embed_dim = esm_embed_dim
        self.hidden_dim = hidden_dim
        self.pooling_strategy = pooling_strategy
        self.num_classes = num_classes
        # ^^ saves the values as attributes of the object!
        
        # Optional projection layer (useful if ESM dim is very large)
        self.use_projection = esm_embed_dim > 512  # Project if embedding is too large
        if self.use_projection:
            self.projection = nn.Linear(esm_embed_dim, hidden_dim)
            self.proj_dropout = nn.Dropout(dropout * 0.5)  # Light dropout for projection
            classifier_input_dim = hidden_dim
        else:
            self.projection = None
            classifier_input_dim = esm_embed_dim
# basically if the neural network dimensional vectors are too large we're going to 
# have a mini network truncate it to something smaller that we can work with
        
        
        # Pooling layer
        if pooling_strategy == "attention":
            self.pooling = AttentionPooling(classifier_input_dim)
            # classifier_input_dim is either hidden_dim or esm_embed_dim if there is no projection
        elif pooling_strategy in ["mean", "max"]:
            self.pooling = None  # Use functional pooling
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        # define our self.pooling mechanism - for our project we're using "attention"
        #sets up how the model will convert the sequence of embedings into a single
        #summary vector

        # Classification head
        """  Each of the output_dim neurons:
  1. Takes all input_dim numbers as input
  2. Has input_dim weights (one weight per input
  connection)
  3. Has 1 bias value
  4. Computes: output = sum(weights × inputs) + bias
  
answer1: 
part a: at the first neuron it takes the four dimensional embedding
for each position and, the neural itself has 4 numbers in a vector,
aka weights. this neuron computes the dimensional embedding multiplied
by the weights and adds the answer in order to get one output.
so you have like vector v multiplied by e the embedding and then you
sum the result to get one output.

part b: now that you have the attention scores wouldn't you multiply 
the vector by the weight's of the last neuron?

answer2:
a concrete pattern that the two step version could learn is like
"if dimension one is high and dimension 4 is low, then this output
occurs, or this relationship affects this, etc."

answer3: 480x256 weight parameters because each of the 256 neurons has 480 weights/
there is 1 bias per neuron so 256 neurons, so it's 481*256
part b: you multiply the pooled vector by w and then sum each element you get, adding the bias b at the end

answer4: i honestly forgot what ReLU means.

answer5: after dropout is applied i know that some neurons are ignored

part b: i have no idea

answer6: after applying sigmoid i have no idea i just know its 
squashed between 0 and 1
part b: the probability would be lower
part c: i have no idea

answer7:
part a: no
part b: yes. i have no idea
part c: less data, always projecting down to something smaller means
truncation of data


  """
        # Build classification head based on config
        # Support both single int and list of ints for CLASSIFIER_HIDDEN_DIM
        if isinstance(hidden_dim, int):
            # Default 3-layer architecture with //2 reduction
            layer_dims = [hidden_dim, hidden_dim // 2]
        elif isinstance(hidden_dim, list):
            # Custom architecture from config
            layer_dims = hidden_dim
        else:
            raise ValueError(f"CLASSIFIER_HIDDEN_DIM must be int or list, got {type(hidden_dim)}")

        # Build sequential layers dynamically
        layers = []
        current_dim = classifier_input_dim

        for i, next_dim in enumerate(layer_dims):
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.ReLU())
            # Apply dropout with decreasing rate for deeper layers
            dropout_rate = dropout if i == 0 else dropout * 0.5
            layers.append(nn.Dropout(dropout_rate))
            current_dim = next_dim

        # Final output layer
        layers.append(nn.Linear(current_dim, num_classes))

        self.classifier = nn.Sequential(*layers) # nn.Sequential creates a container that runs layers in order, one after another
            # what happens if u don't have this?
            # pytorch won't know to put the output of the first layer into the second layer, and so on
        
        # Initialize weights
        self._init_weights() # defined later in the class
        # runs immediately after creating ESMClassifier object
        # sets the intial values for all the weights in the neural network layers
        # instead of purely random, uses Xavier/Glorot which is a smarter initialization strategy
        # it does it to all the neural networks! 
        # this is because in __init__ we did
            # self.projection, self.pooling, self.classifier, and python's __setattr__ method automatically registers these as child modules

        print(f"   ESM Classifier initialized:")
        print(f"   ESM embed dim: {esm_embed_dim}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Pooling: {pooling_strategy}")
        print(f"   Dropout: {dropout}")
        print(f"   Use projection: {self.use_projection}")

        print(f"   Parameters: {sum(p.numel() for p in self.parameters()):,}")
        # self.parameters gets all weight/bias tensors in the model
        # p.numel() counts the elements in each tensor
        # sum adds everything together to see how many parameters we have for training  
    
    """
    for module in self.modules():

    This recursively walks through the tree of registered
    modules:

    ESMClassifier
    ├─ self.projection (nn.Linear)
    ├─ self.pooling (AttentionPooling)
    │   └─ self.attention (nn.Sequential)
    │       ├─ nn.Linear(input_dim, 128)
    │       ├─ nn.Tanh()
    │       └─ nn.Linear(128, 1)
    └─ self.classifier (nn.Sequential)
        ├─ nn.Linear(classifier_input_dim, hidden_dim)
        ├─ nn.ReLU()
        ├─ nn.Dropout()
        ├─ nn.Linear(hidden_dim, hidden_dim // 2)
        ├─ nn.ReLU()
        ├─ nn.Dropout()
        └─ nn.Linear(hidden_dim // 2, num_classes)
    

    
    """

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules(): # self.modules() is how pytorch recursively finds every module/layer including nested ones
            if isinstance(module, nn.Linear): # is the module a nn.Linear layer?
                # if it's not nn.Linear we don't want to initialize layers that don't have weights lol
                nn.init.xavier_uniform_(module.weight)
                # Xavier/Glorot initialization better than pure randomization

                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # initialize biases to 0 


    # where the magic happens during training!
    # takes embeddings from the ESM2 model, shape (16, 200, 480) for example.
    # mask says which positions are padding vs real. has shape (16, 200) batch_size and seq length
    # forward will return torch.Tensor - logits which are the prediction scores
    def forward(self, 
                embeddings: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # forward is a special pytorch method
        # this runs when you call the model, like if you did model(data)
        """
            Forward pass.

            Args:
                    embeddings: ESM embeddings (batch_size, seq_len, embed_dim)
                    mask: Attention mask (batch_size, seq_len) - True for real tokens

                Returns:
                    logits: Classification logits (batch_size,
                    num_classes)
        """

        batch_size, seq_len, embed_dim = embeddings.shape # why .shape?
        # unpacks the embeddings into 3 variables 
        # useful for debugging 

        # Optional projection
        if self.projection is not None:
            embeddings = self.projection(embeddings)
            embeddings = self.proj_dropout(embeddings)
        
        # Sequence pooling
        if self.pooling_strategy == "attention":
            pooled = self.pooling(embeddings, mask) # calls AttentionPooling.forward()
            # input: embeddings with shape (batch, seq_len, embed_dim) -> (16, 200, 480)
            # AttentionPooling computes attention scores for each position
            # weights positions by importants then sums them and the 
            # output is pooled shape (batch, embed_dim) , like (16, 480), one summary vector

        elif self.pooling_strategy == "mean":
            if mask is not None:
                # Masked mean pooling
                embeddings_masked = embeddings * mask.unsqueeze(-1).float()
                seq_lengths = mask.sum(dim=1, keepdim=True).float()
                pooled = embeddings_masked.sum(dim=1) / seq_lengths
                # sum positions and divide by number of real tokens

            else:
                pooled = embeddings.mean(dim=1) # simple mean, no masking
        elif self.pooling_strategy == "max":
            # alternatively, take maximum value of each dimension
            if mask is not None:
                # Masked max pooling
                embeddings_masked = embeddings.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                pooled = embeddings_masked.max(dim=1)[0]
            else:
                pooled = embeddings.max(dim=1)[0] # simple max with no maxing
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
            # error handling


        # final step - classification!
        # take the pooled vector, which is the summary vector that we get (16, 200, 480) -> (16, 480)
        # and then put that through the neural network, aka the classifier, built in __init__
        # nn.Linear(128, 1), the final layer, produces the logit which outputs
        # 16 proteins, and 1 logit (raw prediction score) for each protein
        logits = self.classifier(pooled)
        
        return logits # return the raw predictions 
        # this will later be passed to the loss function during training
        # and update weights
        # we'll apply sigmoid to it as wel to get 0 or 1
"""



"""

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss focuses training on hard examples by down-weighting
    easy examples. Useful when you have imbalanced classes.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for rare class (between 0 and 1)
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha # weighting for class imbalance
        self.gamma = gamma # how much should we focus on the harder examples
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:  # alpha_t = class-specific weight
  # If target=1 (rare class): alpha_t = 0.25
  # If target=0 (common class): alpha_t = 0.75
  # Balances the contribution from imbalanced classes, 1)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            loss: Focal loss value
        """
        # Convert logits to probabilities
        inputs = inputs.squeeze()
        # the logits for each protein made into one clean array

        probs = torch.sigmoid(inputs)
        # squashed as a probability

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # computes how far off using Binary Cross-Entropy (the loss)

        # Compute focal weights
        p_t = probs * targets + (1 - probs) * (1 - targets)
        # how correct is the prediction?
        # returns another array of certainties (how correct it easy)
        # if the number is high then it's easy, if the number is lower/closer to 0 it's hard

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # alpha_t = class-specific weight
        # If target=1 (rare class): alpha_t = 0.25
        # If target=0 (common class): alpha_t = 0.75
        # Balances the contribution from imbalanced classes
        
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        # in the end it tells us how much attention we should pay to bce_loss,
        # that's why we multiply it in the next step

        # Apply focal weighting
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()
    '''
      Visualization:

  Computation Graph (Forward):
  weights → embeddings → logits → probs → bce_loss → focal_loss → mean_loss
                                              ↑          ↑           ↑
                                      [0.001, 2.5,   [0.001, 2.5,   0.6265
                                       0.002, 0.003]  0.002, 0.003]

  Gradient Flow (Backward):
  weights ← embeddings ← logits ← probs ← bce_loss ← focal_loss ← mean_loss
                                              ↑          ↑           ↑
                                        (each gets      (each gets   gradient=1.0
                                         its own        distributed
                                         gradient)      gradient)
                                         '''

"""
  Component 1: alpha_t - Class balancing

  - From step 5: This is either 0.25 (for class 1) or 0.75 (for class 0)
  - Purpose: Mild adjustment for class imbalance

  Component 2: (1 - p_t) ** self.gamma - Hard example focusing

  - This is the focusing mechanism
  - self.gamma = 2.0 (from line 402)

  how (1 - p_t) ** gamma works:

  Remember:
  - High p_t (0.9) = easy example (model confident and correct)
  - Low p_t (0.3) = hard example (model struggling)

  For easy examples (p_t = 0.9):
  (1 - 0.9) ** 2 = 0.1 ** 2 = 0.01
  The weight becomes 0.01 - massively down-weighted! Almost ignored!

  For hard examples (p_t = 0.3):
  (1 - 0.3) ** 2 = 0.7 ** 2 = 0.49
  The weight is 0.49 - stays relatively large!

  For very hard examples (p_t = 0.1):
  (1 - 0.1) ** 2 = 0.9 ** 2 = 0.81
  The weight is 0.81 - nearly full weight!

    If gamma = 0: No focusing, behaves like regular BCE
  - (1 - p_t) ** 0 = 1 for all examples (everything weighted equally)

  If gamma = 2 (your config): Moderate focusing
  - Easy examples (p_t=0.9): weight = 0.01
  - Hard examples (p_t=0.3): weight = 0.49

  If gamma = 5: Extreme focusing
  - Easy examples (p_t=0.9): weight = 0.00001 (almost completely ignored!)
  - Hard examples (p_t=0.3): weight = 0.17

"""


def get_loss_function(loss_type: str = None):
    """
    Get the loss function based on configuration.
    
    Args:
        loss_type: Type of loss ("bce", "focal", "weighted_bce")
        
    Returns:
        loss_fn: PyTorch loss function
    """
    loss_type = loss_type or config.LOSS_TYPE
    
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "focal":
        return FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
    elif loss_type == "weighted_bce":
        # Will need to compute class weights from data
        return nn.BCEWithLogitsLoss()  # Placeholder - weights set in training
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def create_esm_classifier(esm_embed_dim: int = None) -> ESMClassifier:
    """
    Create ESM classifier with default configuration.
    
    Args:
        esm_embed_dim: ESM embedding dimension (auto-detected if None)
        
    Returns:
        model: ESMClassifier instance
    """
    if esm_embed_dim is None:
        esm_embed_dim = config.get_esm_embedding_dim()
    
    return ESMClassifier(
        esm_embed_dim=esm_embed_dim,
        hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
        pooling_strategy=config.POOLING_STRATEGY,
        dropout=config.CLASSIFIER_DROPOUT,
        num_classes=1  # Binary classification
    )