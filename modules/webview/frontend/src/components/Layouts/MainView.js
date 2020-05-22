import React from "react";
import { inject, observer } from "mobx-react";
import Loadable from 'react-loadable';

import RouteEditingBar from "components/RouteEditingBar";
import StatusBar from "components/StatusBar";
import Scene from "components/Scene";
import Loader from "components/common/Loader";
import PlaybackControls from "components/PlaybackControls";

const Navigation = Loadable({
    loader: () => import("components/Navigation"),
    loading() {
      return <div>Loading...</div>;
    }
});

@inject("store") @observer
class SceneView extends React.Component {
    render() {
        const { sceneDimension, meters, monitor,
                options, trafficSignal, video, hmi } = this.props.store;

        const sceneHeightOffset = OFFLINE_PLAYBACK ? 40 /* height of playback control */ : 0;
        return (
            <div className="main-view" style={{ height: sceneDimension.height }}>
                <Scene  width={sceneDimension.width}
                        height={sceneDimension.height - sceneHeightOffset}
                        options={options}
                        invisible={false} />
                {options.showRouteEditingBar
                    ? <RouteEditingBar />
                    : <StatusBar meters={meters}
                                 trafficSignal={trafficSignal}
                                 showNotification={!options.showTasks}
                                 showPlanningRSSInfo={options.showPlanningRSSInfo}
                                 monitor={monitor} />}
                {OFFLINE_PLAYBACK && <PlaybackControls />}
                {hmi.inNavigationMode &&
                    <Navigation viewHeight={sceneDimension.height}
                                viewWidth={sceneDimension.width} />}
            </div>
        );
    }
}

@inject("store") @observer
export default class MainView extends React.Component {
    render() {
        const { isInitialized, sceneDimension } = this.props.store;

        if (!isInitialized && !OFFLINE_PLAYBACK) {
            return <Loader height={sceneDimension.height} />;
        } else {
            return <SceneView />;
        }
    }
}

