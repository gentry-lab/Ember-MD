// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, Match, Switch, Show } from 'solid-js';
import WizardLayout from './components/layout/WizardLayout';
import MDStepLoad from './components/steps/MDStepLoad';
import MDStepConfigure from './components/steps/MDStepConfigure';
import MDStepProgress from './components/steps/MDStepProgress';
import MDStepResults from './components/steps/MDStepResults';
import DockStepLoad from './components/steps/DockStepLoad';
import DockStepConfigure from './components/steps/DockStepConfigure';
import DockStepProgress from './components/steps/DockStepProgress';
import DockStepResults from './components/steps/DockStepResults';
import ViewerMode from './components/viewer/ViewerMode';
import ConformStepLoad from './components/steps/ConformStepLoad';
import ConformStepConfigure from './components/steps/ConformStepConfigure';
import ConformStepProgress from './components/steps/ConformStepProgress';
import ConformStepResults from './components/steps/ConformStepResults';
import ScoreStepLoad from './components/steps/ScoreStepLoad';
import ScoreStepProgress from './components/steps/ScoreStepProgress';
import ScoreStepResults from './components/steps/ScoreStepResults';
import { workflowStore } from './stores/workflow';

const App: Component = () => {
  const { state } = workflowStore;

  return (
    <WizardLayout>
      {/* ViewerMode stays mounted (CSS-hidden) to preserve NGL Stage + WebGL context.
          Destroying/recreating the stage on every mode switch causes OOM from re-parsing structures. */}
      <div
        class="h-full w-full min-w-0"
        style={{ display: state().mode === 'viewer' ? 'block' : 'none' }}
      >
        <ViewerMode />
      </div>

      <Switch>
        {/* Dock mode steps */}
        <Match when={state().mode === 'dock' && state().dockStep === 'dock-load'}>
          <DockStepLoad />
        </Match>
        <Match when={state().mode === 'dock' && state().dockStep === 'dock-configure'}>
          <DockStepConfigure />
        </Match>
        <Match when={state().mode === 'dock' && state().dockStep === 'dock-progress'}>
          <DockStepProgress />
        </Match>
        <Match when={state().mode === 'dock' && state().dockStep === 'dock-results'}>
          <DockStepResults />
        </Match>

        {/* MD mode steps */}
        <Match when={state().mode === 'md' && state().mdStep === 'md-load'}>
          <MDStepLoad />
        </Match>
        <Match when={state().mode === 'md' && state().mdStep === 'md-configure'}>
          <MDStepConfigure />
        </Match>
        <Match when={state().mode === 'md' && state().mdStep === 'md-progress'}>
          <MDStepProgress />
        </Match>
        <Match when={state().mode === 'md' && state().mdStep === 'md-results'}>
          <MDStepResults />
        </Match>

        {/* Conform mode steps */}
        <Match when={state().mode === 'conform' && state().conformStep === 'conform-load'}>
          <ConformStepLoad />
        </Match>
        <Match when={state().mode === 'conform' && state().conformStep === 'conform-configure'}>
          <ConformStepConfigure />
        </Match>
        <Match when={state().mode === 'conform' && state().conformStep === 'conform-progress'}>
          <ConformStepProgress />
        </Match>
        <Match when={state().mode === 'conform' && state().conformStep === 'conform-results'}>
          <ConformStepResults />
        </Match>

        {/* X-ray scoring mode steps */}
        <Match when={state().mode === 'score' && state().scoreStep === 'score-load'}>
          <ScoreStepLoad />
        </Match>
        <Match when={state().mode === 'score' && state().scoreStep === 'score-progress'}>
          <ScoreStepProgress />
        </Match>
        <Match when={state().mode === 'score' && state().scoreStep === 'score-results'}>
          <ScoreStepResults />
        </Match>

      </Switch>
    </WizardLayout>
  );
};

export default App;
