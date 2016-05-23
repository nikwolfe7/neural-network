package mlsp.cs.cmu.edu.dnn.training;

import java.util.Collections;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.util.DNNUtils;

public class MNISTDataInstanceFactory implements DataInstanceFactory {

	String data = "." + DNNUtils.sep + "data" + DNNUtils.sep;
	List<DataInstance> training;
	List<DataInstance> testing;
	
	public MNISTDataInstanceFactory() {
		System.out.println("Reading MNIST data... ");
		DataReader reader = new ReadCSVDataOneHotVectors();
		this.training = reader.getDataFromFile(data + "mnist-train.csv", 784, 10);
		this.testing = reader.getDataFromFile(data + "mnist-test.csv", 784, 10);
		Collections.shuffle(training);
		Collections.shuffle(testing);
	}

	@Override
	public List<DataInstance> getTrainingInstances() {
		return training;
	}

	@Override
	public List<DataInstance> getTestingInstances() {
		return testing;
	}

}
