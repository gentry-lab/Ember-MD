// Copyright (c) 2026 Ember Contributors. MIT License.
/* @refresh reload */
import { render } from 'solid-js/web';
import App from './App';
import './styles.css';
import '../shared/types/electron-api'; // Import for global type augmentation

const root = document.getElementById('root');

if (!root) {
  throw new Error('Root element not found');
}

render(() => <App />, root);
