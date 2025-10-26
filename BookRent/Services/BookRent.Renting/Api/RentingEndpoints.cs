namespace BookRent.Renting.Api;

public static class RentingEndpoints
{
    public static void MapRentingEndpoints(this IEndpointRouteBuilder app)
    {
        RouteGroupBuilder group = app.MapGroup("/renting");
        group.MapGet("/allRentedBooks", GetRentedBooks);
        group.MapGet("/rentHistory/{userId:guid}", GetRentHistory);
        group.MapPost("/rentBooks", RentBooks);
        group.MapPost("/returnBooks", ReturnBooks);
    }
    
    private static IResult GetRentedBooks(HttpRequest request)
    {
        
        return TypedResults.Ok("success");
    }
    private static IResult GetRentHistory(HttpRequest request)
    {
        
        return TypedResults.Ok("success");
    }
    
    private static IResult ReturnBooks(HttpRequest request)
    {
        
        return TypedResults.Ok("success");
    }
    
    private static IResult RentBooks(HttpRequest request)
    {
        
        return TypedResults.Ok("success");
    }
}