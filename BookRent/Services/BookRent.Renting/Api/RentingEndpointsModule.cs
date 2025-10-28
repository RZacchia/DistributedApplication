namespace BookRent.Renting.Api;

internal static class RentingEndpointsModule
{
    internal static void MapRentingEndpoints(this IEndpointRouteBuilder app)
    {
        RouteGroupBuilder group = app.MapGroup("/renting");
        group.MapGet("/allRentedBooks", RentingHistoryEndpoints.GetRentedBooks);
        group.MapGet("/rentHistory/{userId:guid}", RentingHistoryEndpoints.GetRentHistory);
        group.MapPost("/rentBooks", RentingActionsEndpoints.RentBooks);
        group.MapPost("/returnBooks", RentingActionsEndpoints.ReturnBooks);
    }
    
}