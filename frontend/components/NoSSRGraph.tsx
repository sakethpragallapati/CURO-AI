import { useEffect, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

export default function NoSSRGraph(props: any) {
  const fgRef = useRef<any>(null);

  useEffect(() => {
    if (fgRef.current) {
      // Moderate link distance for structure without feeling scattered
      fgRef.current.d3Force('link').distance(90);
      
      // Tight repulsion to ensure bounding boxes don't overlap without sending nodes flying
      fgRef.current.d3Force('charge').strength(-600);

      fgRef.current.d3Force('center').strength(0.08);

      // Pass the instance back to the parent if a callback is provided
      if (props.onRef) {
        props.onRef(fgRef.current);
      }
    }
  }, [props.graphData, props.onRef]);

  return <ForceGraph2D ref={fgRef} {...props} />;
}
