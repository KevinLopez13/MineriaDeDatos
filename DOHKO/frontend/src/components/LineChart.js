// yarn add @nivo/core @nivo/line
import React from 'react';
import { ResponsiveLine } from '@nivo/line';

const data = [
    {
      "id": "japan",
    //   "color": "hsl(290, 70%, 50%)",
      "data": [
        {
          "x": "moto",
          "y": 139
        },
        {
          "x": "bicycle",
          "y": 57
        },
        {
          "x": "horse",
          "y": 215
        },
        {
          "x": "skateboard",
          "y": 196
        },
        {
          "x": "others",
          "y": 274
        }
      ]
    }
  ]

export default function LineChart({line}){

        return (
            <div style={{height:'350px', width:'auto', textAlign:"center"}}>
                {/* <h6>{line.title}</h6> */}
                <ResponsiveLine
                    data={line.datas}
                    margin={{ top: 10, right: 20, bottom: 50, left: 60 }}
                    xScale={{ type: 'point' }}
                    yScale={{
                        type: 'linear',
                        min: 'auto',
                        max: 'auto',
                        stacked: false,
                        reverse: false
                    }}
                    yFormat=" >-.2f"
                    axisTop={null}
                    axisRight={null}
                    axisBottom={{
                        orient: 'bottom',
                        tickSize: 5,
                        tickPadding: 5,
                        tickRotation: 0,
                        legend: line.x_legend,
                        legendOffset: 36,
                        legendPosition: 'middle'
                    }}
                    axisLeft={{
                        orient: 'left',
                        tickSize: 5,
                        tickPadding: 5,
                        tickRotation: 0,
                        legend: line.y_legend,
                        legendOffset: -50,
                        legendPosition: 'middle'
                    }}
                    pointSize={10}
                    pointColor={{ theme: 'background' }}
                    pointBorderWidth={2}
                    pointBorderColor={{ from: 'serieColor' }}
                    pointLabelYOffset={-12}
                    useMesh={true}
                    legends={[]}
                />
            </div>
    );
}