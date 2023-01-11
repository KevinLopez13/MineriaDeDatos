import React from 'react'
// yarn add @nivo/core @nivo/bar
import { ResponsiveBar } from '@nivo/bar'

export default function Bar({ hists=[] }){

    return(
        <div className='row row-cols-1 row-cols-md-3 g-4 justify-content-center'>
        {hists.map((hist, index) => (
            <div id={index} style={{height:'350px', width:'300px', textAlign:"center"}}>
                <h6>{hist.title}</h6>
                <ResponsiveBar
                    data={hist.data}
                    keys={["value"]}
                    indexBy="id"
                    margin={{ top: 5, right: 0, bottom: 100, left: 50 }}
                    layout={hist.layout}
                    groupMode="grouped"
                    valueScale={{ type: 'linear' }}
                    indexScale={{ type: 'band', round: true }}
                    colors={{ scheme: 'nivo' }}
                    colorBy="indexValue"
                    defs={[]}
                    fill={[]}
                    borderColor={{
                        from: 'color',
                        modifiers: [
                            [
                                'darker',
                                1.6
                            ]
                        ]
                    }}
                    axisTop={null}
                    axisRight={null}
                    axisBottom={{
                        tickSize: 10,
                        tickPadding: 10,
                        tickRotation: -45,
                        legend: '',
                        legendPosition: 'middle',
                        legendOffset: 0
                    }}
                    axisLeft={{
                        tickSize: 5,
                        tickPadding: 5,
                        tickRotation: 0,
                        legend: '',
                        legendPosition: 'middle',
                        legendOffset: -40
                    }}
                    enableLabel={false}
                    labelSkipWidth={12}
                    labelSkipHeight={12}
                    labelTextColor={{
                        from: 'color',
                        modifiers: [
                            [
                                'darker',
                                1.6
                            ]
                        ]
                    }}
                    animate={false}
                    legends={[]}
                    role="application"
                    ariaLabel="Nivo bar chart demo"
                    barAriaLabel={function(e){return e.id+": "+e.formattedValue+" in country: "+e.indexValue}}
                />
            </div>
        ))}
        </div>
    )
}