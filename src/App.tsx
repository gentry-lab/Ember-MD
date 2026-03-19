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
import FepScoringPanel from './components/viewer/FepScoringPanel';
import MapMode from './components/map/MapMode';
import ConformStepLoad from './components/steps/ConformStepLoad';
import ConformStepConfigure from './components/steps/ConformStepConfigure';
import ConformStepProgress from './components/steps/ConformStepProgress';
import ConformStepResults from './components/steps/ConformStepResults';
import { workflowStore } from './stores/workflow';

const App: Component = () => {
  const { state } = workflowStore;

  return (
    <WizardLayout>
      {/* ViewerMode stays mounted (CSS-hidden) to preserve NGL Stage + WebGL context.
          Destroying/recreating the stage on every mode switch causes OOM from re-parsing structures. */}
      <div
        class="h-full"
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

        {/* Score mode (FEP scoring, single view) */}
        <Match when={state().mode === 'score'}>
          <FepScoringPanel />
        </Match>

        {/* Map mode (pocket mapping) */}
        <Match when={state().mode === 'map'}>
          <MapMode />
        </Match>
      </Switch>
    </WizardLayout>
  );
};

export default App;
