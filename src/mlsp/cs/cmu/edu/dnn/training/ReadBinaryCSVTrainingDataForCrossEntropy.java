package mlsp.cs.cmu.edu.dnn.training;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class ReadBinaryCSVTrainingDataForCrossEntropy implements DataReader {

	@Override
	public List<DataInstance> getDataFromFile(String fileName, int inputDim, int outputDim) {
		 List<DataInstance> data = new ArrayList<DataInstance>();
		    try {
		      Scanner scn = new Scanner(new File(fileName));
		      while(scn.hasNextLine()) {
		        String[] line = scn.nextLine().split("\\s+|,");
		        double[] arr = new double[line.length];
		        for(int i = 0; i < line.length; i++)
		          arr[i] = Double.parseDouble(line[i]);
		        DataInstance d = new DataInstance(inputDim, outputDim, arr); 
		        double[] out = d.getOutputTruthValue();
		        if(out[0] == 1)
		        	out = new double[] {1,0};
		        else
		        	out = new double[] {0,1};
		        d.replaceOutputTruthValue(out);
		        data.add(d);
		      }
		      scn.close();
		    } catch (FileNotFoundException e) {
		      e.printStackTrace();
		    }
		    return data;
	}

}
