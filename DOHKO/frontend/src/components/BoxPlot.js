import React from "react";
// yarn add react-plotly.js ploty.js

import Plot from "react-plotly.js";

export default function BoxPlot ({boxes=[]}) {

    return(
        <div className='row row-cols-1 row-cols-md-3 g-4 justify-content-md-center'>
        {boxes.map((box, index) => (
            <div id={index} className="mx-4" style={{height:'250px', width:'250px', textAlign:"center"}}>
                <Plot
                    data={[
                        {
                            type: 'box',
                            x: box.data,
                            name: '',
                            marker: {
                                color: 'rgb(163,73,164)',
                            },
                            boxpoints: 'Outliers'
                        },
                    ]}

                    layout={ {
                        width: 250,
                        height: 250,
                        title: box.name,
                        margin:{
                            t:40,
                            b:40,
                            l:0,
                            r:0,
                        },
                        showlegend: false
                    } }

                    config={{
                        scrollZoom: false,
                        modeBarButtonsToRemove: ['zoom2d', 'zoomIn2d', 'zoomOut2d'],
                        displaylogo: false,
                        displayModeBar: false,
                    }}
                    
                />
            </div> 
        ))}
        </div>
    );
}