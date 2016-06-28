package mlsp.cs.cmu.edu.dnn.factory;

import java.util.List;

import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.DNNTrainingModule;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.training.MNISTAltSmallDataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.training.MNISTSmallDataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.util.DNNUtils;
import mlsp.cs.cmu.edu.dnn.util.MaxBinaryThresholdOutput;

public class WeightInitSigmoidDNNFactory extends WeightInitializedDNNFactory {

	public WeightInitSigmoidDNNFactory(DataInstance example, boolean biasComesFirst, String... matrixFiles) {
		super(example, biasComesFirst, matrixFiles);
	}
	
	@Override
	public FeedForwardDNNAbstractFactory getFFDNNFactory(DataInstance example, int... hiddenLayerDimenions) {
		return new SigmoidNetworkFFDNNFactory(example, hiddenLayerDimenions);
	}

	public static void main(String[] args) {
		String[] matrixFiles = {"data" + DNNUtils.sep + "mat-100x401.csv", "data" + DNNUtils.sep + "mat-10x101.csv"};
		DataInstanceFactory dataInstanceFactory = new MNISTAltSmallDataInstanceFactory();
		List<DataInstance> training = dataInstanceFactory.getTrainingInstances();
		List<DataInstance> testing = dataInstanceFactory.getTrainingInstances();
		DNNFactory factory = new WeightInitSigmoidDNNFactory(training.get(0), true, matrixFiles);
		NeuralNetwork net = factory.getInitializedNeuralNetwork();
		DNNTrainingModule testingModule = new DNNTrainingModule(net, training, testing);
		testingModule.setOutputOn(true);
		testingModule.setOutputAdapter(new MaxBinaryThresholdOutput());
		testingModule.doTestTrainedNetwork();
	}

}
