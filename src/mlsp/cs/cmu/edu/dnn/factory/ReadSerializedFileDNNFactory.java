package mlsp.cs.cmu.edu.dnn.factory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;

public class ReadSerializedFileDNNFactory implements DNNFactory {

	private NeuralNetwork network;
	
	public ReadSerializedFileDNNFactory(String fileName) {
		readObject(new File(fileName));
	}
	
	public ReadSerializedFileDNNFactory(File file) {
		readObject(file);
	}
	
	private void readObject(File file) {
		FileInputStream inputStream;
		ObjectInputStream objectStream;
		try {
			inputStream = new FileInputStream(file);
			objectStream = new ObjectInputStream(inputStream);
			this.network = (NeuralNetwork) objectStream.readObject();
			objectStream.close();
			inputStream.close();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			System.out.println("Could not find a required class...");
			e.printStackTrace();
		}
	}
	
	@Override
	public NeuralNetwork getInitializedNeuralNetwork() {
		return network;
	}


}
