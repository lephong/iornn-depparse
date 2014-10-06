DATA_PATH=../../data/wsj-dep/universal/data
K=10
OUTPUT_PATH=experiment

java -classpath ".:lib/trove.jar" -Xmx32g -Djava.io.tmpdir=./ \
	mstparser.DependencyParser \
	test model-name:$OUTPUT_PATH/wsj-univ-2ndorder.model order:2 \
	test-file:$DATA_PATH/dev.conll testing-k:$K output-file:$OUTPUT_PATH/dev-$K-best-mst2ndorder.conll

java -classpath ".:lib/trove.jar" -Xmx32g -Djava.io.tmpdir=./ \
	mstparser.DependencyParser \
	test model-name:$OUTPUT_PATH/wsj-univ-2ndorder.model order:2 \
	test-file:$DATA_PATH/test.conll testing-k:$K output-file:$OUTPUT_PATH/test-$K-best-mst2ndorder.conll

