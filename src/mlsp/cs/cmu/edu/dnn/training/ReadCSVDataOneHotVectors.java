package mlsp.cs.cmu.edu.dnn.training;

import java.util.List;

public class ReadCSVDataOneHotVectors implements DataReader {

	private DataReader reader = new ReadCSVTrainingData();

	@Override
	public List<DataInstance> getDataFromFile(String fileName, int inputDim, int outputDim) {
		List<DataInstance> data = reader.getDataFromFile(fileName, inputDim, 1);
		for (DataInstance instance : data) {
			int label = (int) instance.getOutputTruthValue()[0];
			double[] oneHot = new double[outputDim];
			oneHot[label] = 1;
			instance.replaceOutputTruthValue(oneHot);
		}
		return data;
	}

}
