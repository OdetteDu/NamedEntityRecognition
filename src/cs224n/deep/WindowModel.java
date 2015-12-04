package cs224n.deep;

import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;

import java.text.*;
import java.io.*;

public class WindowModel implements ObjectiveFunction {
	public static final int MAX_ITER = 10;
	public static final double COST_THRESHOLD = 0.01;
	private static int attempt = 1;
	
	private HashMap<String, String> baselineWordMap;
	protected SimpleMatrix L, W1, W2;
	private int wordVectorSize;
	private int hiddenSize;
	
	private int windowSize;
	private int paddingSize;
	
	// Hyperparameter
	private double lambda = 10;
	private double alpha = 0.003;
	
	public WindowModel(int _windowSize, int _hiddenSize, double _lr, double _reg) {
		this.baselineWordMap = new HashMap<String, String>();
		this.L = FeatureFactory.allVecs;
	//	this.L = SimpleMatrix.random(FeatureFactory.allVecs.numRows(),FeatureFactory.allVecs.numCols(), -2, 2, new Random());
		this.wordVectorSize = L.numCols();
		this.hiddenSize = _hiddenSize;
		this.alpha = _lr;
		this.lambda = _reg;
		
		this.windowSize = _windowSize;
		this.paddingSize = _windowSize / 2;
		this.outputMatrixToFile("wordVector.txt", L);
	}

	/**
	 * Initializes the weights randomly.
	 */
	public void initWeights() {
		int X = windowSize * wordVectorSize;
		int H = hiddenSize; 
		int K = Datum.POSSIBLE_LABELS.length;
		double range = Math.sqrt(6) / Math.sqrt(X + H + 1);
		W1 = SimpleMatrix.random(X + 1, H, -range, range, new Random()); //251 * 100
		range = Math.sqrt(6) / Math.sqrt(H + 1+ K);
		W2 = SimpleMatrix.random(H + 1,K, -range, range, new Random()); //101 * 5
	}

	/**
	 * Simplest SGD training
	 */
	public void train(List<Datum> trainData) {
	//	this.baselineTrain(trainData);
		this.nnTrain(trainData);
	}

	public void test(List<Datum> testData) {
	// 	this.baselineTest(testData);
		this.nnTest(testData);
	}
	
	@Override
	public double valueAt(SimpleMatrix label, SimpleMatrix input) {
		
		SimpleMatrix Z1 = input.mult(W1); //1 * 100
		SimpleMatrix H1 = getTanh(Z1); //1 * 101
		SimpleMatrix O = H1.mult(W2); //1 * 5
		SimpleMatrix P = getSoftmax(O); //1 * 5
		
		double p = label.dot(P);
		
		return -Math.log(p);
	}

	private void baselineTrain(List<Datum> trainData) {
		for (Datum datum : trainData) {
			this.baselineWordMap.put(datum.word, datum.label);
		}
	}

	private void baselineTest(List<Datum> testData) {

		List<Prediction> output = new ArrayList<Prediction>();
		for (Datum datum : testData) {
			String word = datum.word;
			String predictedLabel;
			if (this.baselineWordMap.containsKey(word)) {
				predictedLabel = this.baselineWordMap.get(word);
			} else {
				predictedLabel = Datum.DEFAULT_LABEL;
			}
			output.add(new Prediction(word, datum.label, predictedLabel));
		}
		this.outputToFile("baseline.out", output);
	}

	private void nnTrain(List<Datum> trainData)
	{
		System.out.println("W1: "+W1.numRows()+" * "+W1.numCols());
		System.out.println("W2: "+W2.numRows()+" * "+W2.numCols());
		System.out.println("alpha: "+ alpha +" labmda: "+lambda);
		List<Sentence> trainSentences = getSentences(trainData);
		
		int iter = 0;
		while (iter < MAX_ITER) {
			double cost = 0;
			int numInstance = 0;
			for ( Sentence sentence : trainSentences) {
				List<Datum> paddedWords = addPadding(sentence.getWords());
				
				for (int index=paddingSize; index<paddedWords.size()-paddingSize; index++) {
					Datum[] wordsInWindow = new Datum[windowSize];
					
					for (int i=0; i<windowSize; i++) {
						int offset = i-paddingSize; // [-paddingSize,paddingSize]
						wordsInWindow[i] = paddedWords.get(index+offset);
					}
					
					//Forward Pass
					SimpleMatrix X = getWordVector(wordsInWindow); //1 * 251
					SimpleMatrix Z1 = X.mult(W1); //1 * 100
					SimpleMatrix H1 = getTanh(Z1); //1 * 101
					SimpleMatrix O = H1.mult(W2); //1 * 5
					SimpleMatrix P = getSoftmax(O); //1 * 5
					
					//Back Propagation
					SimpleMatrix Y = getY(paddedWords.get(index).label); //1 * 5
					SimpleMatrix delta2 = P.minus(Y); //1 * 5
					SimpleMatrix W2Prime = H1.transpose().mult(delta2); //101 * 5
					
					SimpleMatrix delta1 = delta2.mult(W2.transpose()).elementMult(getDTanh(Z1)); //1 * 101
					delta1 = removeExtraRow(delta1); //1 * 100
					SimpleMatrix W1Prime = X.transpose().mult(delta1); //251 * 100
					
					SimpleMatrix XPrime = delta1.mult(W1.transpose()); //1 * 251
					
					//Check
					boolean gradientCheck = false;
					if (gradientCheck) {
						List<SimpleMatrix> weights = new ArrayList<SimpleMatrix>();
						weights.add(W2);
						weights.add(W1);
						weights.add(X);
						List<SimpleMatrix> matrixDerivatives = new ArrayList<SimpleMatrix>();
						matrixDerivatives.add(W2Prime);
						matrixDerivatives.add(W1Prime);
						matrixDerivatives.add(XPrime);
						boolean isCorrect = GradientCheck.check(Y, weights, matrixDerivatives, this);
						if(!isCorrect) {
							System.out.println("Gradient is wrong!");
							for (int i=0; i<windowSize; i++) {
								System.out.println(wordsInWindow[i].toString()) ;
							}
						//	System.exit(-1);
						}
					}
					
					//Update weights
					boolean regularize = true;
					if(regularize) {
						addRegularization(W2Prime, W2); 
						addRegularization(W1Prime, W1); 
					}
					W2 = W2.minus(W2Prime.scale(alpha)); //101 * 5
					W1 = W1.minus(W1Prime.scale(alpha)); //251 * 100
					updateWordVector(wordsInWindow, XPrime);
					
					// Update cost
					cost += -Math.log(Y.dot(P));
					numInstance++;
				}
			}
			cost /= numInstance;
			iter++;
			System.out.println("Iteration: " + iter + ", Cost: " + cost + ", Instances: " + numInstance);
			
			// Check Convergence
			if (cost < COST_THRESHOLD) break;
		}
		
		System.out.println("Finished Training");
	}

	private void nnTest(List<Datum> testData)
	{
		List<Prediction> output = new ArrayList<Prediction>();
		List<Sentence> testSentences = getSentences(testData);
		for ( Sentence sentence : testSentences) {
			List<Datum> paddedWords = addPadding(sentence.getWords());
			
			for (int index=paddingSize; index<paddedWords.size()-paddingSize; index++) {
				Datum[] wordsInWindow = new Datum[windowSize];
				
				for (int i=0; i<windowSize; i++) {
					int offset = i-paddingSize; // [-paddingSize,paddingSize]
					wordsInWindow[i] = paddedWords.get(index+offset);
				}
				Datum word = wordsInWindow[wordsInWindow.length/2];
				
				//Forward Pass
				SimpleMatrix X = getWordVector(wordsInWindow); //1 * 251
				SimpleMatrix Z1 = X.mult(W1); //1 * 100
				SimpleMatrix H1 = getTanh(Z1); //1 * 101
				SimpleMatrix O = H1.mult(W2); //1 * 5
				SimpleMatrix P = getSoftmax(O); //1 * 5
				
				//Predict
				String label = getLabel(P);
				output.add(new Prediction(word.word, word.label, label));
			}
		}

		this.outputToFile("nn-"+ alpha + "-" + lambda + "-" + attempt +".out", output);
	}

	private List<Sentence> getSentences(List<Datum> data) {
		List<Sentence> sentences = new ArrayList<Sentence>();
		Sentence sentence = new Sentence();
		for(Datum datum : data) {
			if(datum.word.equals(Datum.SEPARATE_WORD)) {
				if(sentence.getWords().size() > 0) {
					sentences.add(sentence);
					sentence = new Sentence();
				}
			}
			else {
				sentence.addWord(datum);
			}			
		}
		
		return sentences;
	}
	
	private List<Datum> addPadding(List<Datum> words) {
		List<Datum> paddedWords = new ArrayList<Datum>();
		Datum start = new Datum(Datum.START_WORD, Datum.DEFAULT_LABEL);
		Datum end = new Datum(Datum.END_WORD, Datum.DEFAULT_LABEL);
		for (int i=0;i<paddingSize;i++) {
			paddedWords.add( start);
		}
		
		for (int i=0;i<words.size();i++) {
			paddedWords.add(words.get(i));
		}
		
		for (int i=0;i<paddingSize;i++) {
			paddedWords.add(end);
		}

		return paddedWords;
	}
	
	private String getLabel(SimpleMatrix p)
	{
		double max = 0;
		String label = null;
		for(int i=0; i<p.numCols(); i++)
		{
			if(p.get(0, i) > max)
			{
				max = p.get(0, i);
				label = Datum.POSSIBLE_LABELS[i];
			}
		}
		return label;
	}

	private SimpleMatrix getWordVector(Datum[] wordsInWindow)
	{
		double[] finalVector = new double[wordVectorSize * wordsInWindow.length + 1];
		for (int i=0; i<wordsInWindow.length; i++)
		{
			double[] wordVector = this.getWordVector(wordsInWindow[i].word);
			for (int j=0; j<wordVector.length; j++)
			{
				finalVector[wordVectorSize * i + j] = wordVector[j];
			}
		}
		//Bias term
		finalVector[finalVector.length-1] = 1;
		double [][] matrixData = {finalVector};
		
		return new SimpleMatrix(matrixData);
	}

	private double[] getWordVector(String word)
	{
		int index;
		if(FeatureFactory.wordToNum.containsKey(word))
		{
			index = FeatureFactory.wordToNum.get(word);
		}
		else
		{
			index = FeatureFactory.NON_EXISTING_VOCAB_INDEX;
		}

		return getRow(L, index);
	}
	
	private void addRegularization(SimpleMatrix wPrime, SimpleMatrix w) {
		SimpleMatrix wCopy = w.copy();
		for (int i=0;i<w.numCols();i++) {
			wCopy.set(wCopy.numRows()-1, i, 0);
		}
		
		wPrime = wPrime.plus(lambda, wCopy);
	}
	
	private void updateWordVector(Datum[] wordsInWindow, SimpleMatrix vector)
	{
		int vectorIndex = 0;
		for (int i=0; i<wordsInWindow.length; i++)
		{
			int rowIndex;
			if (FeatureFactory.wordToNum.containsKey(wordsInWindow[i].word))
			{
				rowIndex = FeatureFactory.wordToNum.get(wordsInWindow[i].word);
			}
			else
			{
				rowIndex = FeatureFactory.NON_EXISTING_VOCAB_INDEX;
			}
			for (int j=0; j<this.wordVectorSize; j++)
			{
				double originValue = L.get(rowIndex, j);
				L.set(rowIndex, j, originValue - alpha * vector.get(0,vectorIndex));
				vectorIndex ++;
			}
		}
	}

	private SimpleMatrix getY(String label)
	{
		double[] y = new double[Datum.POSSIBLE_LABELS.length];
		for (int i=0; i<y.length; i++)
		{
			if(label.equals(Datum.POSSIBLE_LABELS[i]))
			{
				y[i] = 1; 
			}
		}
		double[][] yData = {y};
		return new SimpleMatrix(yData);
	}

	private SimpleMatrix getTanh(SimpleMatrix input)
	{
		double[] tanh = new double[input.numCols() + 1];
		for (int i=0; i<input.numCols(); i++)
		{
			tanh[i] = Math.tanh(input.get(0, i));
		}
		tanh[tanh.length-1] = 1;
		double[][] tanHData = {tanh};
		return new SimpleMatrix(tanHData);
	}

	private SimpleMatrix getDTanh(SimpleMatrix input)
	{
		double[] tanh = new double[input.numCols() + 1];
		for (int i=0; i<input.numCols(); i++)
		{
			tanh[i] = Math.tanh(input.get(0, i));
			tanh[i] = 1 - tanh[i] * tanh[i];
		}
		tanh[tanh.length-1] = 1;
		double[][] tanHData = {tanh};
		return new SimpleMatrix(tanHData);
	}

	private SimpleMatrix getSoftmax(SimpleMatrix input) //?
	{
		double[] softMax = new double[input.numCols()];
		double max = 0;
		for(int i=0; i<input.numCols(); i++)
		{
			if(input.get(0, i) > max)
			{
				max = input.get(0, i);
			}
		}
		double sum = 0;
		for(int i=0; i<input.numCols(); i++)
		{
			softMax[i] = Math.exp(input.get(0, i) - max);
			sum += softMax[i];
		}
		double[] output = new double[input.numCols()];
		for(int i=0; i<softMax.length; i++)
		{
			output[i] = softMax[i]/sum;
		}
//		double checkSum = 0;
//		for(int i=0; i<output.length; i++)
//		{
//			checkSum +=output[i];
//		}
//		System.out.println(checkSum);
		double[][] softMaxData = {output};
		return new SimpleMatrix(softMaxData);
	}

	private SimpleMatrix removeExtraRow(SimpleMatrix input)
	{
		double[] output = new double[input.numCols() - 1];
		for (int i=0; i<input.numCols() - 1; i++)
		{
			output[i] = input.get(0, i);
		}
		double[][] outputData = {output};
		return new SimpleMatrix(outputData);
	}

	private double[] getRow(SimpleMatrix sm, int row)
	{
		double[] vector = new double[sm.numCols()];
		for (int i=0; i<vector.length; i++)
		{
			vector[i] = sm.get(row, i);
		}
		return vector;
	}

	private double[] getCol(SimpleMatrix sm, int col)
	{
		double[] vector = new double[sm.numRows()];
		for (int i=0; i<vector.length; i++)
		{
			vector[i] = sm.get(i, col);
		}
		return vector;
	}

	public void outputToFile(String fileName, List<Prediction> predictions) {

		String outputContent = "";
		for (Prediction prediction : predictions)
		{
			outputContent += prediction + "\n";
		}

		try 
		{
			File file = new File(fileName);
			if (!file.exists()) 
			{
				file.createNewFile();
			}
			BufferedWriter bw = new BufferedWriter(new FileWriter(file));
			bw.write(outputContent);
			bw.close();
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
		}
	}
	
	public void outputMatrixToFile(String fileName, SimpleMatrix sm) {
		
		String outputContent = "";
		for (int i=0; i<sm.numRows(); i++)
		{
			for (int j=0; j<sm.numCols(); j++)
			{
				outputContent += sm.get(i, j) + "";
			}
			outputContent += "\n";
		}

		try 
		{
			File file = new File(fileName);
			if (!file.exists()) 
			{
				file.createNewFile();
			}
			BufferedWriter bw = new BufferedWriter(new FileWriter(file));
			bw.write(outputContent);
			bw.close();
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
		}
	}
}
