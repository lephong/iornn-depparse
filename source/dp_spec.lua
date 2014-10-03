ROOT_LABEL	= 'ROOT'
NUM_THREADS	= 1

-- capital feature
ALL_LOWER = 1
ALL_UPPER = 2
FIRST_UPPER = 3
NOT_FIRST_UPPER = 4
N_CAP_FEAT = 4

-- direction
DIR_L 	= 1
DIR_R 	= 2

-- distance feature
DIST_1		= 1
DIST_2		= 2
DIST_3_6	= 3
DIST_7_INF	= 4
N_DIST_FEAT	= 4


-- dpiornn settings
IORNN_CONTEXT_SIZE	= 3

-- depstruct settings
DEPSTRUCT_N_DEPS = 50

-- for training
MIN_OCCURS_THRESHOLD	= 3
TRAIN_LAMBDA		= 1e-4
TRAIN_LAMBDA_L		= 1e-10
TRAIN_BATCHSIZE		= 32
TRAIN_MAX_N_EPOCHS	= 100
TRAIN_UPDATE_L		= true

TRAIN_WEIGHT_LEARNING_RATE	= 0.1
TRAIN_VOCA_LEARNING_RATE	= 0.1

-- for eval
EVAL_TOOL_PATH	= '../tools/eval-dep.pl'
EVAL_EMAIL_ADDR	= 'lephong.xyz@gmail.com'

K		= 9
alpha	= 0.68

--K_range	= {1,10} -- to choose K [for development], set nil when testing
--alpha_range = {0,1} -- to choose alpha (alpha * mstscore + (1-alpha) * iornnscore) [for development] set nil when testing
punc = true -- taking punc into account for evaluation or not

