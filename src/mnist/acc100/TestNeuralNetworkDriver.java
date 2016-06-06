package mnist.acc100;

import java.util.List;

import mlsp.cs.cmu.edu.dnn.factory.DNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.ReadSerializedFileDNNFactory;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.DNNTrainingModule;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.training.MNISTAltSmallDataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.training.MNISTDataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.training.MNISTSmallDataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.util.DNNUtils;
import mlsp.cs.cmu.edu.dnn.util.MaxBinaryThresholdOutput;
import mlsp.cs.cmu.edu.dnn.util.OutputAdapter;

public class TestNeuralNetworkDriver {

	public static void main(String[] args) {
		/* Test */
		String netFile = "models" + DNNUtils.sep + "mnist-acc100-in-400-out-10-struct-50-50-id-1.dnn";
		OutputAdapter adapter = new MaxBinaryThresholdOutput();
		DataInstanceFactory dataInstanceFactory = new MNISTAltSmallDataInstanceFactory();
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
