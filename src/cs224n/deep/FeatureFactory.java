package cs224n.deep;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import org.ejml.simple.*;


public class FeatureFactory {

	public static final int NON_EXISTING_VOCAB_INDEX = 0;

	private FeatureFactory() {

	}

	static List<Datum> trainData;
	/** Do not modify this method **/
	public static List<Datum> readTrainData(String filename) throws IOException {
		if (trainData==null) trainData= read(filename);
		return trainData;
	}

	static List<Datum> testData;
	/** Do not modify this method **/
	public static List<Datum> readTestData(String filename) throws IOException {
		if (testData==null) testData= read(filename);
		return testData;
	}

	private static List<Datum> read(String filename)
			throws FileNotFoundException, IOException {
		List<Datum> data = new ArrayList<Datum>();
		BufferedReader in = new BufferedReader(new FileReader(filename));
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			String word;
			String label;
			if (line.trim().length() == 0) 
			{
				word = Datum.SEPARATE_WORD;
				label = Datum.DEFAULT_LABEL;
			}
			else
			{
				String[] bits = line.split("\\s+");
				word = bits[0];
				label = bits[1];

				if(word.equalsIgnoreCase("-DOCSTART-")) 
				{
					continue; // Ignore document boundary
				}
			}

			Datum datum = new Datum(word, label);
			data.add(datum);
		}

		return data;
	}


	// Look up table matrix with all word vectors as defined in lecture with dimensionality n x |V|
	static SimpleMatrix allVecs; //access it directly in WindowModel
	public static SimpleMatrix readWordVectors(String vecFilename) throws IOException {
		if (allVecs!=null) return allVecs;

		List<double[]> vectList = new ArrayList<double[]>();
		BufferedReader in = new BufferedReader(new FileReader(vecFilename));
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			String[] numList = line.trim().split("\\s+");

			double[]  vector = new double[numList.length];
			for (int i=0;i<numList.length;i++) {
				vector[i] = Double.parseDouble(numList[i]);
			}
			vectList.add(vector);
		}

		allVecs = new SimpleMatrix(vectList.size(),vectList.get(0).length);
		for (int i=0;i<vectList.size();i++) {
			double[] vector = vectList.get(i);
			for(int j=0;j<vector.length;j++) {
				allVecs.set(i, j, vector[j]);
			}
		}

		return allVecs;
	}

	// might be useful for word to number lookups, just access them directly in WindowModel
	public static HashMap<String, Integer> wordToNum = new HashMap<String, Integer>(); 
	public static HashMap<Integer, String> numToWord = new HashMap<Integer, String>();

	public static HashMap<String, Integer> initializeVocab(String vocabFilename) throws IOException {

		BufferedReader in = new BufferedReader(new FileReader(vocabFilename));
		int index = 0;
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			String word = line.trim();
			if (word.length() == 0) {
				continue;
			}
			wordToNum.put(word, index);
			numToWord.put(index, word);
			index++;
		}
		return wordToNum;
	}









}
