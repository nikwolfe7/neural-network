package mlsp.cs.cmu.edu.dnn.training;

import java.util.Collections;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.util.DNNUtils;

public class MNISTSmallDataInstanceFactory implements DataInstanceFactory {

	String data = "." + DNNUtils.sep + "data" + DNNUtils.sep;
	List<DataInstance> training;
	List<DataInstance> testing;
	
	public MNISTSmallDataInstanceFactory() {
		System.out.println("Reading MNIST data... ");
		DataReader reader = new ReadCSVDataOneHotVectors();
		this.training = reader.getDataFromFile(data + "mnist-small-train.csv", 400, 10);
		this.testing = reader.getDataFromFile(data + "mnist-small-test.csv", 400, 10);
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