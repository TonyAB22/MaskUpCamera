import * as tf from '@tensorflow/tfjs';
import {
  bundleResourceIO,
  cameraWithTensors,
} from '@tensorflow/tfjs-react-native';
import { Camera } from 'expo-camera';
import React, { useEffect, useRef, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';

const TensorCamera = cameraWithTensors(Camera);

const OUTPUT_TENSOR_WIDTH = 270;
const OUTPUT_TENSOR_HEIGHT = 480;

export default function App() {
  const [tfReady, setTfReady] = useState(false);
  const [model, setModel] = useState();
  const [isMask, setIsMask] = useState(null);

  const rafId = useRef(null);


  useEffect(() => {
    async function prepare() {
      rafId.current = null;


      await Camera.requestCameraPermissionsAsync();


      await tf.ready();


      const modelJson = require('./model/model.json');
      const modelWeights = require('./model/weights.bin');

      const model = await tf.loadLayersModel(
        bundleResourceIO(modelJson, modelWeights)
      );
      setModel(model);

      // Ready!!
      setTfReady(true);
    }

    prepare();
  }, []);

  // This will be called when the component in unmounted.
  useEffect(() => {
    return () => {
      if (rafId.current != null && rafId.current !== 0) {
        cancelAnimationFrame(rafId.current);
        rafId.current = 0;
      }
    };
  }, []);


  const handleCameraStream = (images) => {
    console.log('camera ready!');

    const loop = () => {

      if (rafId.current === 0) {
        return;
      }


      tf.tidy(() => {
  
        const imageTensor = images.next().value.expandDims(0).div(127.5).sub(1);

      
        const f =
          (OUTPUT_TENSOR_HEIGHT - OUTPUT_TENSOR_WIDTH) /
          2 /
          OUTPUT_TENSOR_HEIGHT;
        const cropped = tf.image.cropAndResize(
          imageTensor,
          tf.tensor2d([f, 0, 1 - f, 1], [1, 4]),
          // The first box above
          [0],
          // The final size after resize.
          [224, 224]
        );

        // Feed the processed tensor to the model and get result tensor(s).
        const result = model.predict(cropped);
        // Get the actual data (an array in this case) from the result tensor.
        const logits = result.dataSync();
        // Logits should be the probability of two classes (hot dog, not hot dog).
        if (logits) {
          setIsMask(logits[0] > logits[1]);
        } else {
          setIsMask(null);
        }
      });

      rafId.current = requestAnimationFrame(loop);
    };

    loop();
  };

  if (!tfReady) {
    return (
      <View style={styles.loadingMsg}>
        <Text>Loading...</Text>
      </View>
    );
  } else {
    return (
      <View style={styles.container}>
        <TensorCamera
          style={styles.camera}
          autorender={true}
          type={Camera.Constants.Type.front}
          // Output tensor related props.
          // These decide the shape of output tensor from the camera.
          resizeWidth={224}
          resizeHeight={224}
          resizeDepth={3}
          onReady={tensors => handleCameraStream(tensors)}
        />
        <View
          style={
            isMask
              ? styles.resultContainerMask
              : styles.resultContainerNotMask
          }
        >
          <Text style={styles.resultText}>
            {isMask ? 'Mask On' : 'Wear Mask!'}
          </Text>
        </View>
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    position: 'relative',
  },
  camera: {
    width: '100%',
    height: '100%',
    zIndex: 1,
  },
  loadingMsg: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },
  resultContainerMask: {
    position: 'absolute',
    top: 0,
    left: 0,
    zIndex: 100,
    padding: 22,
    borderRadius: 8,
    backgroundColor: '#00aa00' ,
  },
  resultContainerNotMask: {
    position: 'absolute',
    top: 0,
    left: 0,
    zIndex: 100,
    padding: 22,
    borderRadius: 8,
    backgroundColor: '#aa0000',
  },
  resultText: {
    fontSize: 30,
    color: 'white',
  },
});
