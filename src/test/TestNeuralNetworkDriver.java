package test;

import java.util.List;

import mlsp.cs.cmu.edu.dnn.factory.DNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.ReadSerializedFileDNNFactory;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.DNNTrainingModule;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.training.MNISTDataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.util.DNNUtils;
import mlsp.cs.cmu.edu.dnn.util.MaxBinaryThresholdOutput;
import mlsp.cs.cmu.edu.dnn.util.OutputAdapter;

public class TestNeuralNetworkDriver {

	static String modelName = "mnist-test-in-784-out-10-struct-28-28-id-1.dnn";
	
	public static void main(String[] args) {
		/* Test */
		String netFile = "models" + DNNUtils.sep + modelName;
		OutputAdapter adapter = new MaxBinaryThresholdOutput();
		DataInstanceFactory dataInstanceFactory = new MNISTDataInstanceFactory();
		List<DataInstance> testing = dataInstanceFactory.getTestingInstances();

		System.out.println("De-serializing the network..");
		DNNFactory factory = new ReadSerializedFileDNNFactory(netFile);
		NeuralNetwork net = factory.getInitializedNeuralNetwork();
		DNNTrainingModule trainingModule = new DNNTrainingModule(net, testing);
		trainingModule.setOutputOn(false);
		trainingModule.setOutputAdapter(adapter);
		trainingModule.doTestTrainedNetwork();
	}

}
